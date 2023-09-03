from monai.transforms import (
                            Compose,
                            OneOf,
                            SomeOf,

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
    def __init__(self, all_key, input_key, input_size:Sequence[int], aug_Lv:int=0) -> None:
        
        
        # preprocesing cfg
        intensity_cfg = EasyDict()
        
        intensity_cfg.CT_min = -1000
        intensity_cfg.CT_max = 1000
        intensity_cfg.CT_clip = True
        intensity_cfg.PET_min = 0
        intensity_cfg.PET_max = 40  # 'PET_max' is for test_transform
        intensity_cfg.PET_max2 = 20
        intensity_cfg.PET_clip = False
        
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
            augmentation_cfg.gamma_prob = 0.3
            augmentation_cfg.gamma_range = (0.7, 1.5)
        
        # lv.2
        if aug_Lv > 1: 
            augmentation_cfg.low_resolution_prob = 0.2
            augmentation_cfg.low_resolution_min_ration = 0.4
            augmentation_cfg.noise_prob = 0.2
            augmentation_cfg.blur_prob = 0.2
            augmentation_cfg.sigma_range = (0.25, 1.1)
            augmentation_cfg.shift_prob = 0.2
            augmentation_cfg.shift_offsets = (-0.1, 0.1)
        
        
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
            LoadImaged(keys=self.all_key, ensure_channel_first=True),
            EnsureTyped(keys=self.all_key, track_meta=False),
            Orientationd(keys=self.all_key, axcodes='RAS'),
            ScaleIntensityRanged(keys='ct',
                                    a_min=intensity_cfg.CT_min, a_max=intensity_cfg.CT_max,
                                    b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys='pet',
                                    a_min=intensity_cfg.PET_min, a_max=intensity_cfg.PET_max,
                                    b_min=0, b_max=1, clip=intensity_cfg.PET_clip),
            CropForegroundd(keys=self.all_key, source_key='pet'), # source_key 'ct' or 'pet'
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
        
        if self.aug_Lv == 2 :
            aug_list = self._get_lv1_auglist(augmentation_cfg) + self._get_lv2_auglist(augmentation_cfg)
    
        if self.aug_Lv == 3 : 
            aug_list = self._get_lv1_auglist(augmentation_cfg) + self._get_lv2_auglist(augmentation_cfg) + self._get_lv3_auglist(augmentation_cfg)
            
        
        if not is_randAug:
            return Compose(default_aug + aug_list)
        
        n = len(aug_list)
        chosen = SomeOf(
            transforms=aug_list,
            num_transforms=(n//2, n*9//10)  # 0.5 ~ 0.9
        )
        
        return Compose(default_aug + chosen)
        
        
        
    def _get_default_auglist(self, intensity_cfg)->list:
        # intensity_cfg.img_size
        return [
            LoadImaged(keys=self.all_key, ensure_channel_first=True),
            EnsureTyped(keys=self.all_key, track_meta=True), # for training track_meta=False, monai.data.set_track_meta(false)
            Orientationd(keys=self.all_key, axcodes='RAS'),
            ScaleIntensityRanged(keys='ct',
                                 a_min=intensity_cfg.CT_min, a_max=intensity_cfg.CT_max,
                                 b_min=0, b_max=1, clip=True),
            OneOf([
                ScaleIntensityRanged(keys='pet',
                                    a_min=intensity_cfg.PET_min, a_max=intensity_cfg.PET_max,
                                    b_min=0, b_max=1, clip=intensity_cfg.PET_clip),
                ScaleIntensityRanged(keys='pet',
                                    a_min=intensity_cfg.PET_min, a_max=intensity_cfg.PET_max2,
                                    b_min=0, b_max=1, clip=intensity_cfg.PET_clip),    
            ]),
            CropForegroundd(keys=self.all_key, source_key='pet'), # source_key 'ct' or 'pet'
            OneOf([
                RandCropByPosNegLabeld(keys=self.all_key, label_key='label', 
                                       spatial_size=self.img_size, 
                                       pos=1, neg=0.2, num_samples=1,
                                       image_key='pet',
                                       image_threshold=0), # 흑색종일때 label에 따라서 잘 되는지 확인해야함 
                RandSpatialCropd(keys=self.all_key, roi_size=self.img_size, random_size=False
                                 )],
                weights=[0.8, 0.2]
                )
        ]
        
    def _get_lv1_auglist(self, augmentation_cfg):
            return [
            # Lv 1. nnUNet impl Aug
            # flip + rotation + elastic + scale(brightness) + contrast(gamma)
            RandFlipd(keys=self.all_key, prob=augmentation_cfg.flip_prob, spatial_axis=0), 
            RandFlipd(keys=self.all_key, prob=augmentation_cfg.flip_prob, spatial_axis=1),
            RandFlipd(keys=self.all_key, prob=augmentation_cfg.flip_prob, spatial_axis=2),           
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
        
    def _get_lv2_auglist(self, augmentation_cfg=None):
        H, W, D = (2.03642,  2.03642, 3.)
        ratio =  augmentation_cfg.low_resolution_min_ration
        offsets = [(round(np.random.uniform(0,H*ratio),3), round(np.random.uniform(0,D*ratio),3)) for _ in range(100)]
        return [
            # if you use low_resolution aug, track_meta = True
            OneOf(
                [
                    Spacingd(keys=self.input_key,
                             pixdim=(H, W, D), mode=("bilinear")),
                    Compose([OneOf([Spacingd(keys=self.input_key,
                                    pixdim=(2.03642 + HW_offset,  2.03642 + HW_offset, 3. + D_offset), mode=("bilinear"))
                                    for HW_offset, D_offset in offsets]),
                             Spacingd(keys=self.input_key,
                                      pixdim=(H, W, D), mode=("bilinear")),
                             Resized(keys=self.input_key, 
                                     spatial_size=self.img_size)
                            ]),
                ],
                weights=[1. -augmentation_cfg.low_resolution_prob, augmentation_cfg.low_resolution_prob] # 30% of low resolution
            ),
            OneOf([RandGaussianNoised(keys=self.input_key, prob=augmentation_cfg.noise_prob, 
                                  mean=np.random.uniform(0.0, 0.1) if np.random.random() > 0.5 else 0 , 
                                  std=np.random.uniform(0.001, 0.1)) for i in range(100)]),
            RandGaussianSmoothd(keys=self.input_key, prob=augmentation_cfg.blur_prob,
                                sigma_x=augmentation_cfg.sigma_range, sigma_y=augmentation_cfg.sigma_range,  sigma_z=augmentation_cfg.sigma_range),
            RandShiftIntensityd(keys=self.input_key, prob=augmentation_cfg.shift_prob,
                                offsets=augmentation_cfg.shift_offsets)
        ]

            
    def _get_lv3_auglist(self, augmentation_cfg=None):
        return
    
        # ratio = np.random.uniform(low=augmentation_cfg.low_resolution_low,
        #                           high=augmentation_cfg.low_resolution_high)
        
        # low_resol_size = [int(size * ratio) for size in self.input_size]
        
        # return [OneOf[
        #     Compose()
        # ]
        #     Resized(keys=self.all_key, 
        #             spatial_size=low_resol_size,),
        #     RandZoomd(keys=self.all_key, prob=augmentation_cfg.zoom_prob,
        #               min_zoom=augmentation_cfg.min_zoom,
        #               max_zoom=augmentation_cfg.max_zoom) 
        #     ] + [
        #         OneOf([RandGaussianNoised(keys=self.input_key[idx], prob=augmentation_cfg.noise_prob,
        #                               mean=np.random.uniform(0.0, augmentation_cfg.noise_mean) if np.random.rand() > 0.3 else 0,
        #                               std= np.random.uniform(0.0, augmentation_cfg.noise_std)) for _ in range(100)])
        #         for idx in range(len(self.input_key))
        #     ]  + [
        #         OneOf([RandGaussianSmoothd(keys=self.input_key[idx], prob=augmentation_cfg.blur_prob,
        #                                    sigma_x=augmentation_cfg.sigma_range, sigma_y=augmentation_cfg.sigma_range, sigma_z=augmentation_cfg.sigma_range)])
        #         for idx in range(len(self.input_key))
        #     ] + [
        #         RandShiftIntensityd(keys=self.input_key[idx], prob=augmentation_cfg.shift_prob,
        #                             offsets=augmentation_cfg.shift_offsets, safe=True) 
        #         for idx in range(len(self.input_key))
        #     ]

         
        
    
        
        
        
        
        
        
        