from models.rcrnet_vit import RCRNet_vit
from models.rcrnet_res2 import RCRNet_res2net

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import cv2

from models.Equirec2Cube import Equirec2Cube
from models.Cube2Equirec import Cube2Equirec

num_cube = 6


class ImgModel(nn.Module):
    '''
        RCRNet+cube maps
    '''
    def __init__(self, output_stride=16):
        super(ImgModel, self).__init__()
        # video mode + video dataset
        self.backbone = RCRNet_vit(
            n_classes=1,
            output_stride=output_stride,
        )
        self.backbone_cube = RCRNet_res2net(
            n_classes=1,
            output_stride=output_stride,
            pretrained=True
        )

        ER_height = 640
        CB_height = ER_height // 2  # bigger than 256
        ER_m_height = 16  # high level features of cube maps branch
        self.equi2cube = Equirec2Cube(1, ER_height, ER_height * 2, CB_height, 90)
        self.cube2equi = Cube2Equirec(1, 10, ER_m_height, ER_m_height * 2)
        self.equi2cube_aux = Equirec2Cube(num_cube, ER_m_height, ER_m_height * 2, 10, 90)
        self.shapeCube = CB_height

        self.gateChannel_ER = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.gateChannel_CB = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.gateSpatial_ER = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.gateSpatial_CB = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.CALayer = ChannelAttention(256)

        if self.training: self.initialize_w_image_pretrain()

    def initialize_w_image_pretrain(self):
            backbone_vit_pretrain = torch.load(os.getcwd() + '/pretrain/RCRNet_vit_pretrain.pth')
            backbone_res2net_pretrain = torch.load(os.getcwd() + '/pretrain/RCRNet_res2_pretrain.pth')

            all_params = {}
            for k, v in self.backbone.state_dict().items():
                if k in backbone_vit_pretrain.keys():
                    v = backbone_vit_pretrain[k]
                    all_params[k] = v
            self.backbone.load_state_dict(all_params)

            all_params_cube = {}
            for k, v in self.backbone_cube.state_dict().items():
                if k in backbone_res2net_pretrain.keys():
                    v = backbone_res2net_pretrain[k]
                    all_params_cube[k] = v
            self.backbone_cube.load_state_dict(all_params_cube)

    def forward(self, frame, ER_frame, ER_gt=None):
        feats = self.backbone.feat_conv(frame)

        #  equirectangular to cube
        cubes_in = self.equi2cube.ToCubeTensor(ER_frame)   # 'back', 'down', 'front', 'left', 'right', 'up'

        # gain high level of cube maps
        cube_feats = self.backbone_cube.feat_conv(cubes_in)
        cube_feat_high = cube_feats[3]
        cubes_bottleneck = self.cube2equi.ToEquirecTensor(cube_feat_high)

        # fuse the ER and CB features at high level
        feats_mutual_er, feats_mutual_cb = self.fusion_mutual_attention(feats[3], cubes_bottleneck)
        feats_bottleneck = feats[3] + feats_mutual_er

        if ER_gt == None:
            pred = self.backbone.seg_conv(feats[0], feats[1], feats[2], feats_bottleneck, frame.shape[2:])

            return pred
        else:
            pred = self.backbone.seg_conv(feats[0], feats[1], feats[2], feats_bottleneck, frame.shape[2:])

            # gain the auxiliary outputs of cube map branch
            feats_aux = cubes_bottleneck + feats_mutual_cb
            feats_aux_cube = self.equi2cube_aux.ToCubeTensor(feats_aux)
            feats_cube_bottleneck = cube_feats[3] + self.CALayer(feats_aux_cube)
            pred_aux = self.backbone_cube.seg_conv(cube_feats[0], cube_feats[1], cube_feats[2], feats_cube_bottleneck,
                                                   [self.shapeCube, self.shapeCube])

            cube_gt = self.equi2cube.ToCubeTensor(ER_gt)

            return pred, pred_aux, cube_gt

        #  debug
        #ER_out = self.cube2equi.ToEquirecTensor(cubes_in)
        #for ii in range(num_cube):
        #    cv2.imwrite(str(ii) + '_cube_in.png', cubes_in[ii, :, :, :].permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('ER_in_1.png', frame[0].permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('ER_in_2.png', frame[1].permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('ER_out_1.png', ER_out[0].permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('ER_out_2.png', ER_out[1].permute(1, 2, 0).cpu().data.numpy() * 255)

    def fusion_mutual_attention(self, feat_er, feat_cb):
        bs, ch, hei, wei = feat_er.size()

        feat_er_flatten = torch.flatten(feat_er, start_dim=2, end_dim=3)  # C*HW
        feat_cb_flatten = torch.flatten(feat_cb, start_dim=2, end_dim=3)  # C*HW

        ChannelAff = torch.matmul(feat_er_flatten, feat_cb_flatten.permute(0, 2, 1))
        ChannelAffAtten1 = F.softmax(ChannelAff, dim=1)  # C*C
        ChannelAffAtten2 = F.softmax(ChannelAff, dim=2)  # C*C

        ChannelAffER = torch.matmul(ChannelAffAtten1, feat_er_flatten)
        ChannelAffER = ChannelAffER.reshape([bs, ch, hei, wei])
        ChannelAffER_gate = self.gateChannel_ER(ChannelAffER)
        ChannelAffER = ChannelAffER.mul(ChannelAffER_gate)

        ChannelAffCB = torch.matmul(ChannelAffAtten2, feat_cb_flatten)
        ChannelAffCB = ChannelAffCB.reshape([bs, ch, hei, wei])
        ChannelAffCB_gate = self.gateChannel_CB(ChannelAffCB)
        ChannelAffCB = ChannelAffCB.mul(ChannelAffCB_gate)

        ch_feat_er_flatten = torch.flatten(ChannelAffER, start_dim=2, end_dim=3)  # C*HW
        ch_feat_cb_flatten = torch.flatten(ChannelAffCB, start_dim=2, end_dim=3)  # C*HW

        SaptialAff = torch.matmul(ch_feat_er_flatten.permute(0, 2, 1), ch_feat_cb_flatten)
        SpatialAffAtten1 = F.softmax(SaptialAff, dim=1)  # HW*HW
        SaptialAffAtten2 = F.softmax(SaptialAff, dim=2)  # HW*HW

        SpatialAffER = torch.matmul(ch_feat_er_flatten, SpatialAffAtten1)
        SpatialAffER = SpatialAffER.reshape([bs, ch, hei, wei])
        SpatialAffER_gate = self.gateSpatial_ER(SpatialAffER)
        SpatialAffER = SpatialAffER.mul(SpatialAffER_gate)

        SpatialAffCB = torch.matmul(ch_feat_cb_flatten, SaptialAffAtten2)
        SpatialAffCB = SpatialAffCB.reshape([bs, ch, hei, wei])
        SpatialAffCB_gate = self.gateSpatial_CB(SpatialAffCB)
        SpatialAffCB = SpatialAffCB.mul(SpatialAffCB_gate)

        return SpatialAffER, SpatialAffCB


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = self.conv1(max_out)
        return self.sigmoid(y) * x
