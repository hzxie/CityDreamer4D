# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Jiaxiang Tang (@ashawkey)
# @Date:   2023-04-15 10:33:32
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-12-27 19:18:33
# @Email:  ashawkey1999@gmail.com
# @Ref: https://github.com/ashawkey/torch-ngp

import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CXX_STD = "-std=c++17" if torch.__version__ >= "2.0" else "-std=c++14"

setup(
    name="grid_encoder",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="grid_encoder_ext",
            sources=[
                "grid_encoder_ext.cu",
                "bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", CXX_STD],
                "nvcc": [
                    "-O3",
                    CXX_STD,
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
