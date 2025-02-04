conda create -n hisup2 python=3.7
conda install conda-forge::gcc_linux-64=10 conda-forge::gxx_linux-64=10 -y
conda install conda-forge::libxcrypt -y  # solves the error when installing the apt module
