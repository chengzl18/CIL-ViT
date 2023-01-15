#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu Nov 22 12:09:27 2018
Info:
'''

import glob

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from helper import RandomTransWrapper

from transformers import AutoFeatureExtractor, ViTFeatureExtractor, DetrFeatureExtractor


class CarlaH5Data():
    def __init__(self,
                 train_folder,
                 eval_folder,
                 batch_size=4, num_workers=4, distributed=False, model_type='CNN'):

        if model_type == 'CNN':
            self.loaders = {
                "train": torch.utils.data.DataLoader(
                    CarlaH5Dataset(
                        data_dir=train_folder,
                        train_eval_flag="train"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=True
                ),
                "eval": torch.utils.data.DataLoader(
                    CarlaH5Dataset(
                        data_dir=eval_folder,
                        train_eval_flag="eval"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=False
                )}
        elif model_type == 'DETR':
            self.loaders = {
                "train": torch.utils.data.DataLoader(
                    CarlaDETRDataset(
                        data_dir=train_folder,
                        train_eval_flag="train"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=True
                ),
                "eval": torch.utils.data.DataLoader(
                    CarlaDETRDataset(
                        data_dir=eval_folder,
                        train_eval_flag="eval"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=False
                )}
        else:
            self.loaders = {
                "train": torch.utils.data.DataLoader(
                    CarlaViTDataset(
                        data_dir=train_folder,
                        model_type=model_type,
                        train_eval_flag="train"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=True
                ),
                "eval": torch.utils.data.DataLoader(
                    CarlaViTDataset(
                        data_dir=eval_folder,
                        model_type=model_type,
                        train_eval_flag="eval"),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=False
                )}



class CarlaH5Dataset(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        if train_eval_flag == 'train':
            self.data_list.remove(data_dir+'data_06790.h5')
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'])[file_idx]
            # print("img shape:", img.shape)
            img = self.transform(img)
            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)
            # 2 Follow lane, 3 Left, 4 Right, 5 Straight
            # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
            command = int(target[24])-2
            # Steer, Gas, Brake (0,1, focus on steer loss)
            target_vec = np.zeros((4, 3), dtype=np.float32)
            target_vec[command, :] = target[:3]
            # in km/h, <90
            speed = np.array([target[10]/90, ]).astype(np.float32)
            mask_vec = np.zeros((4, 3), dtype=np.float32)
            mask_vec[command, :] = 1

            # TODO
            # add preprocess
            # print("img shape:", img.shape)
        return img, speed, target_vec.reshape(-1), \
            mask_vec.reshape(-1),


class CarlaViTDataset(Dataset):
    def __init__(self, data_dir, model_type,
                 train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        if train_eval_flag == 'train':
            self.data_list.remove(data_dir+'data_06790.h5')
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        if model_type == 'ViT':
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                cache_dir='/home3/private/zhanghaoye/packages/transformers/vit-base')
        elif model_type == 'MAE':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                "facebook/vit-mae-base",
                cache_dir='/home3/private/zhanghaoye/packages/transformers/mae-base')

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'])[file_idx]
            # print("img shape:", img.shape)
            img = self.transform(img)

            output = self.feature_extractor(img, return_tensors = 'pt')
            output = output['pixel_values'].squeeze(0)
            # print(output.shape)
            # for key in output:
            #     print(key)
            #     print("feature extractor:", output[key].shape)

            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)
            # 2 Follow lane, 3 Left, 4 Right, 5 Straight
            # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
            command = int(target[24])-2
            # Steer, Gas, Brake (0,1, focus on steer loss)
            target_vec = np.zeros((4, 3), dtype=np.float32)
            target_vec[command, :] = target[:3]
            # in km/h, <90
            speed = np.array([target[10]/90, ]).astype(np.float32)
            mask_vec = np.zeros((4, 3), dtype=np.float32)
            mask_vec[command, :] = 1

            # TODO
            # add preprocess
            # print("img shape:", img.shape)
        return output, speed, target_vec.reshape(-1), \
            mask_vec.reshape(-1),

class CarlaDETRDataset(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        if train_eval_flag == 'train':
            self.data_list.remove(data_dir+'data_06790.h5')
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            "facebook/detr-resnet-50",
            cache_dir='/home3/private/zhanghaoye/packages/transformers/detr-res50',
            do_resize=False,)

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'])[file_idx]
            # print("img shape:", img.shape)
            img = self.transform(img)
            # print("img shape new:", img.shape)

            output = self.feature_extractor(img, return_tensors = 'pt')
            # assert output['pixel_mask'] == torch.tensor([])
            for key in output:
                # print(key)
                output[key] = output[key].squeeze(0)
                if key == 'pixel_mask':
                    for item in torch.where(output[key] == 0):
                        # print(item)
                        assert item.tolist() == []
                # print(key, output[key].shape)
            # for key in output:
            #     print(key)
            #     print("feature extractor:", output[key].shape)

            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)
            # 2 Follow lane, 3 Left, 4 Right, 5 Straight
            # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
            command = int(target[24])-2
            # Steer, Gas, Brake (0,1, focus on steer loss)
            target_vec = np.zeros((4, 3), dtype=np.float32)
            target_vec[command, :] = target[:3]
            # in km/h, <90
            speed = np.array([target[10]/90, ]).astype(np.float32)
            mask_vec = np.zeros((4, 3), dtype=np.float32)
            mask_vec[command, :] = 1

            # print("img shape:", img.shape)
        return output[list(output.keys())[0]], speed, target_vec.reshape(-1), \
            mask_vec.reshape(-1),
