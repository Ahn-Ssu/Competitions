# 4D Block multi-label classification
- 기간: 1월 2일 ~ 1월 30일
- 대회 URL: [DACON - 4D block clf](https://dacon.io/competitions/official/236046/overview/description)
- 성적: 상위 8%

## Problem
- train data: 가상 배경화면에 classification object (4D block)이 놓여져 있는 그림
- test  data: 책상 등 실생활, 실 사용에서 촬영된 4D block 그림 


## approach 
background synthesis: 
1. train data img를 closing을 활용해 foreground의 mask를 획득. 
2. Coco 2017 Validation dataset을 배경으로 추출해내 마스크로 이미지를 합성
3. 합성된 이미지를 통해 학습을 진행, 모델의 generalization을 증가, 학습 성능 개선 


## what I learned 
- [efficientNet architecture](https://arxiv.org/abs/1905.11946): baseline으로 제공된 pretrained model, 대회 시작전 까지 mobileNet, MBConv, Squeeze & Excitation에 대해서 몰랐었음
- [Albumentation](https://albumentations.ai): image data augmentation library. torchVision transforms을 사용하느 것보다 속도가 우수하게 빠른 증강 라이브러리. 
- Mosaic augmentation: 여러장의 이미지를 한 장으로 만드는 증강 기법. 
- [Cosine annealing with warm restart](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html): scheduler 효력을 무시하던 편이었는데, 구글링을 하면서 공부하던 중 발견했음. 첨부된 URL은 PyTorch에서 지원하는 버전으로 ware restart는 구현되어 있지 않다. 그래서 [이 깃허브](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup)에 구현된 버전을 사용했는데 **정상적인 학습 진행이 되지 않아서 끝내 적용에는 실패함.**
- Multi GPU learning libaray: 종류는 많은데, 우선 [이 글](https://medium.com/daangn/pytorch-multi-gpu-학습-제대로-하기-27270617936b)을 읽고 개념을 잡았다. [NVIDIA apex](https://github.com/NVIDIA/apex) library로 첫 시도를 해보려고 한다. 

