# Carcarsh Video Classification
- 기간: 2월 6일 ~ 3월 13일
- 대회 URL: [제1회 코스포 x 데이콘 자동차 충돌 분석 AI경진대회(채용 연계형)](https://dacon.io/competitions/official/236046/overview/description)
- 성적: 

## Problem
- train data: class imblance가 매우 심한 dashcam videos, 실제로는 1 column classification이지만 multi-label classification으로 전환 할 수 있는 lookup table이 주어짐



## approach 
background synthesis: 
1. lookup table을 이용해서 multi class를 multi-label로 4개로 분해 
2. 충돌['crash'] column의 경우, 사고 연루['ego-involve'] column에 종속된다고 판단, 굳이 따로 학습 시킬 필요가 없는 것 같아서 제외
3. 충돌을 제외하고 3가지 속성 (사고 연루, 날씨, 낮/밤)을 예측하는 모델을 각각 만듬 - (우선, multi-label을 한번에 예측하는 모델을 만들 필요가 없다고 생각되었음)
4. StratifiedKFold CV로 5개의 모델을 얻은 뒤 한 카테고리 마다 앙상블
5. 예측한 값을 단순 평균내서 column 별 예측 생성 
6. 사고 연루 컬럼을 기준으로 충돌 컬럼 생성
7. 디코딩 -> 예측 제출

*weather의 경우 labeling이 매우 노이즈 하다는 지적을 대회 내에서 받고 있음
*그 밖에도 데이터 자체가 이상한 경우가 존재하는데, 디테일하게 작업을 수행하지는 않음


## what I learned 
- PyTorchLightning: PyTorch-based code를 추상화할 수 있는 라이브러리, document가 매우 잘 되어 있고 advanced skill 들도 override 형태로 커스텀해서 사용할 수 있음. 특히 multi-GPU 세팅 경우 단순하게 trainer에게서 # of gpu 와 device idx를 제공하면 됨
- [huggingface](https://huggingface.co/): transformer based NLP lib, [scikit-multilearn](http://scikit.ml/): sklearn for multi-label classification, [pytorchvideo](https://pytorchvideo.org/): pytorch lib for video processing, [einops](https://einops.rocks/): easy reshaping lib, [decord](https://github.com/dmlc/decord): fast video reader
- Video augmentation 방법: img 랑 다르게 전체에 고르게 적용되어야 하고, task에 따라서 ㅁ ㅐ우 예민
- sudo labeling: 데이터 셋이 부족할 때, label이 있는 일부 학습데이터를 이용해서 label이 없는 학습데이터에 레이블링을 수행한 뒤에 더 큰 학습데이터 셋을 사용해서 모델의 generalizability를 높이려는 기술 
- deep learning model ensembling: average, majority voting, weighted average, stacking, boosting 등이 있는데 competition의 경우 public score를 이용해서 weighted average를 수행해도 괜찮을 듯 
