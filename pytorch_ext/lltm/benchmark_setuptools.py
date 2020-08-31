from cpp import LLTM as cpp_lltm
from py import LLTM as py_lltm
from cuda_cpp import LLTM as cuda_lltm
from benchmark import benchmark


if __name__ == "__main__":

    benchmark(cuda_lltm, 'CUDA', device='cuda', steps=1000)

    benchmark(cpp_lltm, 'C++', device='cuda', steps=1000)
    benchmark(py_lltm, 'Py ', device='cuda', steps=1000)

    benchmark(cpp_lltm, 'C++', device='cpu', steps=1000)
    benchmark(py_lltm, 'Py ', device='cpu', steps=1000)
