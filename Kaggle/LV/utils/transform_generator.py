from monai.transforms import (
                            Compose,
                            OneOf,
                            
                            AsDiscreted,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            
                            Orientationd,
                            CropForegroundd, 
                            RandCropByPosNegLabeld,
                            RandSpatialCropd,

                            ## nnUNet v1 impl Aug
                            RandFlipd,
                            RandRotated,
                            Rand3DElasticd,
                            RandScaleIntensityd,
                            RandAdjustContrastd, # gamma correction
                            
                            Resized,
                            Spacingd,
                            RandGaussianNoised,
                            RandGaussianSmoothd,
                            RandShiftIntensityd,

                            # RandAdjustContrastd,
                            # RandGaussianSharpend,

                            # RandCoarseDropoutd,
                            # RandCoarseShuffled
                        )

import numpy as np
from easydict import EasyDict
from collections.abc import Sequence




class MONAI_transformerd():
    def __init__(self, all_key, input_key, input_size, aug_Lv:int=0):
        
        
        # preprocesing cfg
        basic_cfg = EasyDict()
        
        basic_cfg.num_class = 3 # bg, WM, cortex
        basic_cfg.MRI_min = 0.
        basic_cfg.MRI_max = 255

        # Augmentation cfg
        augmentation_cfg = EasyDict()
        # lv.1 
        # lv.1 aug list is from nnUNet impl
        if aug_Lv > 0: 
            augmentation_cfg.flip_prob = 0.5
            augmentation_cfg.rotation_prob = 0.2 
            augmentation_cfg.rotation_degree = 15. * (2. * np.pi / 360.)
            augmentation_cfg.elastic_prob = 0.2
            augmentation_cfg.elastic_sigma_range = (5, 13) # if x > 15, it causes CPU bottleneck 
            augmentation_cfg.elastic_magnitude_range = (100, 400) # if x > 500, it makes an unrealistic img
            augmentation_cfg.scaling_prob = 0.2
            augmentation_cfg.scaling_fator = 0.25
            augmentation_cfg.gamma_prob = 0.2
            augmentation_cfg.gamma_range = (0.7, 1.5)
        
        
        self.aug_Lv = aug_Lv # 1
        self.all_key = all_key #'image', 'label'
        self.input_key = input_key # 'image'
        self.label_key = all_key[-1] # this code could be problematic
        self.img_size = input_size # 192x192x192..?
        self.basic_cfg = basic_cfg
        self.augmentation_cfg = augmentation_cfg
        
    def get_CFGs(self):
        return self.basic_cfg, self.augmentation_cfg
    
    def generate_test_transform(self, basic_cfg=None):
        if basic_cfg == None:
            basic_cfg = self.basic_cfg
        
        return Compose([
                    LoadImaged(keys=self.all_key, image_only=True, ensure_channel_first=True),
                    EnsureTyped(keys=self.all_key, device=None, track_meta=False),
                    # Orientationd(keys=self.all_key, axcodes="RAS"),
                    # CropForegroundd(keys=self.all_key, source_key=self.label_key),
                    CropForegroundd(keys=self.all_key, source_key=self.label_key, allow_smaller=False),
                    Resized(keys=self.all_key, spatial_size=self.img_size),
                    AsDiscreted(keys='label', to_onehot=self.basic_cfg.num_class),
                    ScaleIntensityRanged(keys=self.input_key,
                                        a_max=self.basic_cfg.MRI_max, a_min=self.basic_cfg.MRI_min,
                                        b_max=1, b_min=0, clip=True),
                ])
        
    def generate_train_transform(self, 
                                 basic_cfg:EasyDict=None, augmentation_cfg:EasyDict=None, 
                                 is_randAug=False):
        if augmentation_cfg == None:
            augmentation_cfg = self.augmentation_cfg
            
        if basic_cfg == None:
            basic_cfg = self.basic_cfg
            
        
        default_aug = self._get_default_auglist(basic_cfg)
        if self.aug_Lv == 0 :
            return Compose(default_aug)
        
        if self.aug_Lv == 1 :
            aug_list = self._get_lv1_auglist(augmentation_cfg)
        
        
        if not is_randAug:
            return Compose(default_aug + aug_list)
        
        # n = len(aug_list)
        # chosen = SomeOf(
        #     transforms=aug_list,
        #     num_transforms=(n//2, n*9//10)  # 0.5 ~ 0.9
        # )
        
        # return Compose(default_aug + chosen)
        
        
        
    def _get_default_auglist(self, basic_cfg)->list:
        # basic_cfg.img_size
        return [
                    LoadImaged(keys=self.all_key, image_only=True, ensure_channel_first=True),
                    EnsureTyped(keys=self.all_key, device=None, track_meta=False),
                    # Orientationd(keys=self.all_key, axcodes="RAS"),
                    # CropForegroundd(keys=self.all_key, source_key=self.label_key),
                    CropForegroundd(keys=self.all_key, source_key=self.label_key, allow_smaller=False),
                    Resized(keys=self.all_key, spatial_size=self.img_size),
                    AsDiscreted(keys='label', to_onehot=self.basic_cfg.num_class),
                    ScaleIntensityRanged(keys=self.input_key,
                                        a_max=self.basic_cfg.MRI_max, a_min=self.basic_cfg.MRI_min,
                                        b_max=1, b_min=0, clip=True),
                ]
        
        
    def _get_lv1_auglist(self, augmentation_cfg):
            return [
            # Lv 1. nnUNet impl Aug
            # flip + rotation + elastic + scale(brightness) + contrast(gamma)
            RandFlipd(keys=self.all_key, spatial_axis=0),
            RandFlipd(keys=self.all_key, spatial_axis=1),
            RandFlipd(keys=self.all_key, spatial_axis=2),
            RandRotated(keys=self.all_key, prob=augmentation_cfg.rotation_prob,
                        range_x=augmentation_cfg.rotation_degree, range_y=augmentation_cfg.rotation_degree, range_z=augmentation_cfg.rotation_degree),
            Rand3DElasticd(keys=self.all_key, prob=augmentation_cfg.elastic_prob,
                           sigma_range=augmentation_cfg.elastic_sigma_range,
                           magnitude_range=augmentation_cfg.elastic_magnitude_range),
            RandScaleIntensityd(keys=self.input_key, prob=augmentation_cfg.scaling_prob,
                                factors=augmentation_cfg.scaling_fator),
            RandAdjustContrastd(keys=self.input_key, prob=augmentation_cfg.gamma_prob,
                                gamma=augmentation_cfg.gamma_range),
            ]
        
    
         
        
    
        
        
        