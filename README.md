# Pytorch Samples

## Setup steps
```bash
# download and install conda
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda3.sh

# for slient mode, please refer to https://docs.anaconda.com/anaconda/install/silent-mode/
bash ./anaconda3.sh

export PATH=$PATH:~/anaconda3/bin
source ~/.bashrc

# create conda env.
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install pytorch

# CUDA 9.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0
# CUDA 10.0
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
# CUDA 10.1
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# PILLOW 7.0 is not compatible with pytorch 1.3.*, downgrade to 6.1
# ImportError: cannot import name 'PILLOW_VERSION' from 'PIL' 
conda install pillow=6.1

# install git and clone this repo
conda install git
git clone https://github.com/loveunk/pytorch_samples.git

# run the MNIST training and test sample
cd pytorch_samples/mnist/
python main.py
```

## Verify PyTorch GPU
```python
import torch

# Expect `True`
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.version.cuda())
```
