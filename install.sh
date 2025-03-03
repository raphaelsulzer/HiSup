#!/bin/bash
set -e

# Local variables
ENV_NAME=hisup2
PYTHON=3.10.10

# Installation script for Anaconda3 environments
echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath /opt/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath /data/rsulzer/miniconda3`
fi
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo


echo "________________ Conda environment setup _______________"
echo

# Check if the environment exists
if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
    echo "Conda environment '$ENV_NAME' already exists. Removing..."

    # Remove the environment
    conda env remove --name "$ENV_NAME" --yes > /dev/null 2>&1

    # Double-check removal
    if conda env list | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
        echo "Failed to remove the environment '$ENV_NAME'."
        exit 1
    else
        echo "Conda environment '$ENV_NAME' removed successfully."
    fi
fi

## Create a conda environment from yml
echo "Create conda environment '$ENV_NAME'."
conda create -y --name $ENV_NAME python=$PYTHON > /dev/null 2>&1

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}



echo "________________ Package installation _______________"
echo

# pip install -r requirements.txt

conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y
conda install conda-forge::libxcrypt -y  # solves the error when installing the apt module

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cudatoolkit-dev -y
conda install cudatoolkit=11.8 -y
cd hisup/csrc/lib && make && cd ../../../

conda install conda-forge::opencv -y
conda install anaconda::scipy -y
pip install wandb

## install boundary-api
cd ~/python/boundary-iou-api/
pip install -e .
cd ~/remote_python/HiSup

## install lidar_poly
pip install copclib
conda install conda-forge::colorlog -y

## install pointpillars
conda install numba::numba -y
cd ~/python/PointPillars
python setup.py build_ext --inplace
pip install -e .

## now install HiSup

pip install -e .



