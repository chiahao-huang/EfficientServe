from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='adaptive_dynamic_cache_ops',
    ext_modules=[
        CUDAExtension(
            name='adaptive_dynamic_cache_ops',
            sources=['adaptive_dynamic_cache.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

#python adaptive_dynamic_cache_setup.py build_ext --inplace
