import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("diffvoxel", "cuda")
_ext_sources = glob.glob(osp.join(_ext_src_root, "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "*.cu")
)

requirements = ["torch>=1.4"]

exec(open(osp.join("diffvoxel", "_version.py")).read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
setup(
    name="diffvoxel",
    version=__version__,
    author="Craig Lei LI",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="diffvoxel._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
