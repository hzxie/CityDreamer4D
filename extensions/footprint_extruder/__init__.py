# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Haozhe Xie
# @Date:   2023-12-23 11:30:15
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-07-30 20:38:57
# @Email:  root@haozhexie.com

import torch

import footprint_extruder_ext


class FootprintExtruder(torch.nn.Module):
    def __init__(
        self,
        l1_height=0,
        roof_height=1,
        l1_id_offset=0,
        roof_id_offset=1,
        bldg_inst_range=[100, 5000],
    ):
        super(FootprintExtruder, self).__init__()
        self.l1_height = l1_height
        self.roof_height = roof_height
        self.l1_id_offset = l1_id_offset
        self.roof_id_offset = roof_id_offset
        self.bldg_inst_range = bldg_inst_range

    def forward(self, volume, bev_ins, tp_hf, bu_hf):
        return FootprintExtruderFunction.apply(
            volume,
            bev_ins,
            tp_hf,
            bu_hf,
            self.l1_height,
            self.roof_height,
            self.l1_id_offset,
            self.roof_id_offset,
            self.bldg_inst_range,
        )


class FootprintExtruderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        volume,
        bev_ins,
        tp_hf,
        bu_hf,
        l1_height,
        roof_height,
        l1_id_offset,
        roof_id_offset,
        footprint_id_range,
    ):
        # volume.shape: (H, W, D)
        # bev_ins.shape: (H, W)
        return footprint_extruder_ext.forward(
            volume,
            bev_ins,
            tp_hf,
            bu_hf,
            l1_height,
            roof_height,
            l1_id_offset,
            roof_id_offset,
            footprint_id_range[0],
            footprint_id_range[1],
        )
