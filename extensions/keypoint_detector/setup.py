# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Haozhe Xie
# @Date:   2024-11-03 16:38:40
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-11-03 16:40:00
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-fopenmp"]
nvcc_args = []

setup(
    name="keypoint_detector",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "keypoint_detector",
            [
                "bindings.cpp",
                "keypoint_detector_ext.cu",
            ],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
