# WallPapering Long-tailed Multi-class Classification
- 기간: 4월 10일 ~ 5월 22일
- 대회 URL: [도배 하자 유형 분류 AI 경진대회](https://dacon.io/competitions/official/236082/overview/description)
- 성적:

## Problem
* Train Data: Multi-class(19) images 
  *  Imbalanced -- Long-tailed shape
    *  Extremely small class (4 samples)
  *  Impurity -- mis categorized in the train dataset
  *  Ambiguity of the classes 

## approach 
1. Crawling Images - Data Augmentation
  * With Chrome API, Crawling Wallpapering Issue Image 
  * After that, selete valid images 
2. Removing Impurity - Data Clearning 
  * Re-categorize the data samples as correctly as possible in terms of similarity via HUMAN bias
  * Remove some Ambiguity

*class간 우세가 분류에 분명히 존재하는데 주최측에서 대회 중반부에 공개한 cls definition은 별 도움이 안됨 


## what I learned 
- Long-tailed classification에 대해서 처음 알게됨 
