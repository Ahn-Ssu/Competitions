from monai.transforms import (
                            Compose,
                            OneOf,

                            LoadImaged,
                            EnsureTyped,
                            ScaleIntensityRanged,
                            NormalizeIntensityd,
                            HistogramNormalized,
                            
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
        intensity_cfg = EasyDict()
        
        intensity_cfg.Xray_min = 0.
        intensity_cfg.Xray_max = 2**8-1
        intensity_cfg.clip = True
        intensity_cfg.znorm_mean = 0.4690341824996129
        intensity_cfg.znorm_std  = 0.30681989313646496
        
#         hist eq-ed mean and std of C101
#         0.4690341824996129
#         0.30681989313646496

        # Augmentation cfg
        augmentation_cfg = EasyDict()
        # lv.1
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
        
        
        self.aug_Lv = aug_Lv
        self.all_key = all_key
        self.input_key = input_key
        self.img_size = input_size
        self.intensity_cfg = intensity_cfg
        self.augmentation_cfg = augmentation_cfg
        
    def get_CFGs(self):
        return self.intensity_cfg, self.augmentation_cfg
    
    def generate_test_transform(self, intensity_cfg=None):
        if intensity_cfg == None:
            intensity_cfg = self.intensity_cfg
        
        return Compose([
        LoadImaged(keys=self.input_key, image_only=True, ensure_channel_first=True),
        EnsureTyped(keys=self.input_key, device=None, track_meta=False),
        Resized(keys=self.input_key, spatial_size=self.img_size),
        ScaleIntensityRanged(keys=self.input_key,
                             a_max=255.0, a_min=0., b_max=1, b_min=0, clip=True),
        HistogramNormalized(keys=self.input_key),
        NormalizeIntensityd(keys=self.input_key, subtrahend=intensity_cfg.znorm_mean, divisor=intensity_cfg.znorm_std),
    ])
        
    def generate_train_transform(self, 
                                 intensity_cfg:EasyDict=None, augmentation_cfg:EasyDict=None, 
                                 is_randAug=False):
        if augmentation_cfg == None:
            augmentation_cfg = self.augmentation_cfg
            
        if intensity_cfg == None:
            intensity_cfg = self.intensity_cfg
            
        
        default_aug = self._get_default_auglist(intensity_cfg)
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
        
        
        
    def _get_default_auglist(self, intensity_cfg)->list:
        # intensity_cfg.img_size
        return [
                
            LoadImaged(keys=self.input_key, image_only=True, ensure_channel_first=True),
            EnsureTyped(keys=self.input_key, device=None, track_meta=False),
            Resized(keys=self.input_key, spatial_size=self.img_size),
            ScaleIntensityRanged(keys=self.input_key,
                                a_max=255.0, a_min=0., b_max=1, b_min=0, clip=True),
            HistogramNormalized(keys=self.input_key),
            NormalizeIntensityd(keys=self.input_key, subtrahend=intensity_cfg.znorm_mean, divisor=intensity_cfg.znorm_std),
        ]
        
        
    def _get_lv1_auglist(self, augmentation_cfg):
            return [
            # Lv 1. nnUNet impl Aug
            # flip + rotation + elastic + scale(brightness) + contrast(gamma)
#             RandRotated(keys=self.all_key, prob=augmentation_cfg.rotation_prob,
#                         range_x=augmentation_cfg.rotation_degree, range_y=augmentation_cfg.rotation_degree, range_z=augmentation_cfg.rotation_degree),
#             Rand3DElasticd(keys=self.all_key, prob=augmentation_cfg.elastic_prob,
#                            sigma_range=augmentation_cfg.elastic_sigma_range,
#                            magnitude_range=augmentation_cfg.elastic_magnitude_range),
            RandScaleIntensityd(keys=self.input_key, prob=augmentation_cfg.scaling_prob,
                                factors=augmentation_cfg.scaling_fator),
            RandAdjustContrastd(keys=self.input_key, prob=augmentation_cfg.gamma_prob,
                                gamma=augmentation_cfg.gamma_range),
            ]
        
    
         
        
    
        
        
        
        
        
        
        
