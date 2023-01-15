#!/usr/bin/env python
# coding=utf-8
'''
Origin Author:Tai Lei
Date:Thu Nov 22 12:09:33 2018
Info:
'''

import torch
import torch.nn as nn
from transformers import ViTModel, ViTMAEModel, DetrModel


class CarlaNet(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[0]),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[1]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[4]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[5]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[6]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[7]),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
                nn.Linear(8192, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, speed):
        img = self.conv_block(img)
        # print("conv out shape:", img.shape)
        img = img.view(-1, 8192)
        # print("reshape:", img.shape)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([out(emb) for out in self.branches],
                           dim=1)
        pred_speed = self.speed_branch(img)

        return output, pred_speed



class CarlaViT(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaViT, self).__init__()

        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            cache_dir='/home3/private/zhanghaoye/packages/transformers/vit-base')

        self.img_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        outputs = self.model(pixel_values=img)
        img = outputs.pooler_output
        # print("model last hidden shape:", img.shape)
        img = img.view(-1, 768)
        # print("model resize hidden shape:", img.shape)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([out(emb) for out in self.branches],
                           dim=1)
        pred_speed = self.speed_branch(img)

        return output, pred_speed


class CarlaMAE(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaMAE, self).__init__()

        self.model = ViTMAEModel.from_pretrained(
            "facebook/vit-mae-base",
            cache_dir='/home3/private/zhanghaoye/packages/transformers/mae-base')

        self.img_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        outputs = self.model(pixel_values=img)
        img = outputs.last_hidden_state
        # print("model last hidden shape:", img.shape)
        img = torch.mean(img, dim=1)
        # print("mean states:", img.shape)
        img = img.view(-1, 768)
        # print("model resize hidden shape:", img.shape)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([out(emb) for out in self.branches],
                           dim=1)
        pred_speed = self.speed_branch(img)

        return output, pred_speed



class CarlaDETR(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaDETR, self).__init__()

        self.model = DetrModel.from_pretrained(
            "facebook/detr-resnet-50",
            cache_dir='/home3/private/zhanghaoye/packages/transformers/detr-res50')

        self.img_fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        outputs = self.model(pixel_values=img)
        img = outputs.last_hidden_state
        # print("model last hidden shape:", img.shape)
        img = torch.mean(img, dim=1)
        # print("mean states:", img.shape)
        img = img.view(-1, 256)
        # print("model resize hidden shape:", img.shape)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        output = torch.cat([out(emb) for out in self.branches],
                           dim=1)
        pred_speed = self.speed_branch(img)

        return output, pred_speed