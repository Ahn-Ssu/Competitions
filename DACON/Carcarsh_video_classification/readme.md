# Carcarsh Video Classification
- 기간: 2월 6일 ~ 3월 13일
- 대회 URL: [제1회 코스포 x 데이콘 자동차 충돌 분석 AI경진대회(채용 연계형)](https://dacon.io/competitions/official/236046/overview/description)
- 성적: 

## Problem
- train data: class imblance가 매우 심한 dashcam videos, multi-label classification 



## approach 
background synthesis: 
1. train data img를 closing을 활용해 foreground의 mask를 획득. 
2. Coco 2017 Validation dataset을 배경으로 추출해내 마스크로 이미지를 합성
3. 합성된 이미지를 통해 학습을 진행, 모델의 generalization을 증가, 학습 성능 개선 


## what I learned 
- 
