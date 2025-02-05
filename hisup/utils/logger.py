# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging, colorlog, sys, os

#
# def setup_logger(name, save_dir, out_file='log.txt'):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#
#     if save_dir:
#         fh = logging.FileHandler(os.path.join(save_dir, out_file))
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#
#     return logger


def make_logger(name="MyLogger",level=logging.INFO,
                filepath=None):
    """
    Attach a stream handler to all loggers.

    Parameters
    ------------
    level : enum (int)
        Logging level, like logging.INFO
    capture_warnings: bool
        If True capture warnings
    filepath: None or str
        path to save the logfile

    Returns
    -------
    logger: Logger object
        Logger attached with a stream handler
    """

    # make sure we log warnings from the warnings module
    # logging.captureWarnings(capture_warnings)

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [(%(filename)s:%(lineno)3s)] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            # 'INFO': 'green',
            'DEBUG': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        },
        secondary_log_colors={},
        style='%'
    )

    # create a basic formatter
    # formatter = logging.Formatter(formatter)

    # if no handler was passed use a StreamHandler
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = 0

    if not any([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    formatter = logging.Formatter("[%(asctime)s] [(%(filename)s:%(lineno)3s)] [%(levelname)s] %(message)s")

    if filepath and not any([isinstance(handler, logging.FileHandler) for handler in logger.handlers]):
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # set nicer numpy print options
    # np.set_printoptions(precision=3, suppress=True)

    # logger.addHandler(logging.StreamHandler())

    return logger
