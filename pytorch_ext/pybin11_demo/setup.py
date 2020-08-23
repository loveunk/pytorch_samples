# compile command: python setup.py build_ext --inplace

from setuptools import setup, Extension

functions_module = Extension(
    name='callee',
    sources=['callee.cpp'],
    include_dirs=[
        'D:\\Anaconda3\\envs\\open-mmlab\\lib\\site-packages\\pybind11\\include',
        'D:\\Anaconda3\\envs\\open-mmlab\\include']
)

setup(ext_modules=[functions_module])
