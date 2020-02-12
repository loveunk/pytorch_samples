# Pytorch Samples

## Setup steps
```bash
# download and install conda
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

bash ./Anaconda3-2019.10-Linux-x86_64.sh

export PATH=$PATH:~/anaconda3/bin
source ~/.bashrc

# create conda env.
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install pytorch
# please check the cuda version, if not 9.0, modify the following command line
conda install pytorch torchvision cudatoolkit=9.0

# PILLOW 7.0 is not compatible with pytorch 1.3.*, downgrade to 6.1
# ImportError: cannot import name 'PILLOW_VERSION' from 'PIL' 
conda install pillow=6.1

# install git and clone this repo
conda install git
conda clone https://github.com/loveunk/pytorch_samples.git

# run the MNIST training and test sample
cd pytorch_samples/mnist/
python main.py
```