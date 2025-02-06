conda create -n hisup2 python=3.7
conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y
conda install conda-forge::libxcrypt -y  # solves the error when installing the apt module


## install with ptv3

conda create -n pointcept python=3.8 -y
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cudatoolkit-dev -y
conda install cudatoolkit=11.8 -y
conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y
conda install anaconda::cython -y
# now 'hisup/csrc/lib make' should work

## now install flash-attention
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation

## now install HiSup
pip install -r requirements.txt
# install boundary-api
pip install wandb
pip install copclib
conda install conda-forge::colorlog -y


## now install ptv3
conda install conda-forge::addict -y
pip install spconv-cu118
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y

