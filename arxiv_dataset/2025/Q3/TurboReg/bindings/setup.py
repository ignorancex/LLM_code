import torch
import os
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ROOT_DIR = os.path.join(osp.dirname(osp.abspath(__file__)), "..")

include_dirs = [
    osp.join(ROOT_DIR, "./turboreg/include")
]

eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
if eigen_include_dir:
    include_dirs.append(eigen_include_dir)

sources = (
    glob.glob(osp.join(ROOT_DIR, "./turboreg/src", "*.cpp")) + [osp.join(ROOT_DIR, "./bindings/pybinding.cpp")] 
)

has_cuda = torch.cuda.is_available() and len(glob.glob(osp.join("src", "*.cu"))) > 0

ext_modules = []
if has_cuda:
    ext_modules.append(
        CUDAExtension(
            name="turboreg_gpu", 
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],  
                "nvcc": ["-O2"],  
            },
        )
    )
else:
    ext_modules.append(
        CppExtension(
            name="turboreg_gpu", 
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=["-O2", "-std=c++17"],  
        )
    )

setup(
    name="turboreg_gpu",
    version="1.0",  
    author="Shaocheng Yan",  
    author_email="shaochengyan@whu.edu.cn", 
    description="Python bindings for TurboReg using PyTorch and pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},  
    install_requires=["torch"],
)
