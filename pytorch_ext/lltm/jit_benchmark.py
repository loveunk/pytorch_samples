from torch.utils.cpp_extension import load

lltm_cpp = load(name="lltm_cpp", sources=["cpp/lltm.cpp"], verbose=True)

from cpp.lltm import LLTM as cpp_lltm
import time
import torch
from benckmark import benckmark


if __name__ == "__main__":
    benckmark(cpp_lltm, 'C++', device='cuda', steps=10000)
    benckmark(cpp_lltm, 'C++', device='cpu', steps=10000)
