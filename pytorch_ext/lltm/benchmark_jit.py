from torch.utils.cpp_extension import load
from benchmark import benchmark

'''
For windows platform, please do jit compile in x64 Native Tools Command Prompt for VS 
'''

if __name__ == "__main__":
    lltm_cpp = load(name="lltm_cpp", sources=["cpp/lltm.cpp"], verbose=True)
    import cpp
    benchmark(cpp.LLTM, 'C++', device='cuda', steps=1000)
    benchmark(cpp.LLTM, 'C++', device='cpu', steps=1000)

    lltm_cuda = load(name='lltm_cuda', sources=['cuda_cpp/lltm_cuda.cpp', 'cuda_cpp/lltm_cuda_kernel.cu'])
    import cuda_cpp
    benchmark(cuda_cpp.LLTM, 'CUDA', device='cuda', steps=1000)
