import os
import time
import argparse
import logging
import random
import numpy as np
import datetime
import wandb
import os.path as osp
import json
from tqdm import tqdm

from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.dataset import build_train_dataset, build_val_dataset
from hisup.utils.comm import to_single_device
from hisup.solver import make_lr_scheduler, make_optimizer
from hisup.utils.logger import make_logger
from hisup.utils.miscellaneous import save_config
from hisup.utils.metric_logger import MetricLogger
from hisup.utils.checkpoint import DetectronCheckpointer
from hisup.utils.metrics.cIoU import compute_IoU_cIoU

from tools.test_pipelines import generate_coco_ann

from ptv3.model import PointTransformerV3

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k] * loss_dict[k]
                          for k in self.loss_weights.keys()])

        return total_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default="./config-files/lidarpoly_hrnet48_debug.yaml",
                        )

    parser.add_argument("--log-to-wandb",
                        help="Activate logging to weights and biases",
                        type=bool,
                        default=False,
                        )
    
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')

    parser.add_argument("--seed",
                        default=2,
                        type=int)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    
    return args

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_wandb(cfg):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="HiSup image only",
        # track hyperparameters and run metadata
        config=dict(cfg)
    )



def log_loss(meters,epoch_size,max_epoch,epoch,it,maxiter,learning_rate):

    eta_batch = epoch_size * (max_epoch - epoch + 1) - it + 1
    eta_seconds = meters.time.global_avg * eta_batch
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    logger.info(
        meters.delimiter.join(
            [
                "eta: {eta}",
                "epoch: {epoch}/{maxepoch}",
                "iter: {iter}/{maxiter}",
                "{meters}",
                "lr: {lr:.6f}",
                "max mem (GB): {memory:.0f}\n",
            ]
        ).format(
            eta=eta_string,
            epoch=epoch,
            maxepoch=max_epoch,
            iter=it,
            maxiter=maxiter,
            meters=str(meters),
            lr=learning_rate,
            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
        )
    )




def validation(model,val_dataset,device,outfile,gtfile):

    model.eval()
    results = []
    iou = 0.0; ciou = 0.0
    for it, (images, annotations) in enumerate(tqdm(val_dataset, desc="Validation")):
        with torch.no_grad():

            images = images.to(device)
            annotations = to_single_device(annotations, device)
            output, _ = model(images, annotations)
            output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']

        for b in range(batch_size):
            filename = annotations[b]['filename']
            # img_id = int(filename[:-4])
            img_id = int(filename.split('_')[0][5:])

            scores = batch_scores[b]
            polys = batch_polygons[b]

            image_result = generate_coco_ann(polys, scores, img_id)
            if len(image_result) != 0:
                results.extend(image_result)

    if len(results):
        logger.info(f'Writing validation results to {outfile}')
        with open(outfile, 'w') as _out:
            json.dump(results, _out)

        iou, ciou = compute_IoU_cIoU(outfile,gtfile)
    else:
        logger.info(f"No polygons predicted")


    model.train()

    return iou, ciou



def train(cfg):

    logger = logging.getLogger("Training")
    device = cfg.MODEL.DEVICE
    model = BuildingDetector(cfg)
    model = model.to(device)

    pt_model = PointTransformerV3(in_channels=3)

    train_dataset = build_train_dataset(cfg)
    val_dataset, gt_file = build_val_dataset(cfg)
    gt_file = osp.abspath(gt_file)
    
    optimizer = make_optimizer(cfg,model)
    scheduler = make_lr_scheduler(cfg,optimizer)
    
    loss_reducer = LossReducer(cfg)
    
    arguments = {}
    arguments["epoch"] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    checkpointer = DetectronCheckpointer(cfg,
                                        model,
                                        optimizer,
                                        save_dir=cfg.OUTPUT_DIR,
                                        save_to_disk=True,
                                        logger=logger)

    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    epoch_size = len(train_dataset)

    global_iteration = epoch_size*start_epoch

    if cfg.LOG_TO_WANDB:
        setup_wandb(cfg)

    best_iou = 0.0

    for epoch in range(start_epoch+1, arguments['max_epoch']+1):
        meters = MetricLogger(" ")
        model.train()
        arguments['epoch'] = epoch

        for it, (images, points, annotations) in enumerate(train_dataset):

            a=5
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_single_device(annotations,device)
            
            loss_dict, _ = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                meters.update(loss=loss_reduced, **loss_dict_reduced)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_iteration +=1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if it % 20 == 0 or it+1 == len(train_dataset):
                log_loss(meters,
                         epoch_size,max_epoch,epoch,
                         it,len(train_dataset),
                         optimizer.param_groups[0]["lr"])

            # if it % 200 == 0 and it > 0:
            #     break

        outfile = osp.join(cfg.OUTPUT_DIR,'validation','validation_{:05d}.json'.format(epoch))
        os.makedirs(osp.dirname(outfile),exist_ok=True)
        iou, ciou = validation(model, val_dataset, device, outfile=outfile, gtfile=gt_file)

        if cfg.LOG_TO_WANDB:
            # make wandb dict
            wandb_dict = {}
            for key in meters.meters.keys():
                if 'loss' in key:
                    wandb_dict[key] = meters.meters[key].global_avg
            wandb_dict['val_iou'] = iou
            wandb_dict['val_ciou'] = ciou
            wandb.log(wandb_dict)

        checkpointer.save('model_{:05d}'.format(epoch))
        if iou > best_iou:
            logger.info(f"New best IoU of {iou}")
            best_iou = iou
            checkpointer.save('model_best')

        scheduler.step()

    wandb.finish()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

if __name__ == "__main__":

    pycocotools_logger = logging.getLogger("pycocotools")
    pycocotools_logger.propagate = False  # Prevents log messages from propagating
    pycocotools_logger.handlers.clear()  # Removes existing handlers
    pycocotools_logger.setLevel(logging.CRITICAL)  # Suppresses all logs

    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.LOG_TO_WANDB = args.log_to_wandb

    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))

    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        if os.path.isdir(output_dir) and args.clean:
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    logger = make_logger('Training', filepath=osp.join(output_dir,'train.log'))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file,"r") as cf:
        config_str = "\n" + cf.read()
        logger.debug(config_str)

    logger.debug("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)
    set_random_seed(args.seed, True)
    train(cfg)

