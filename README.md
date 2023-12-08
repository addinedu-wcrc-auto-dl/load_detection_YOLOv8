# 보행약자를 위한 노면인식 시스템

## 사용한 데이터셋

AI Hub의 베리어프리존(장애물 없는 생활공간) 주행영상 ([https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=186](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=186))

- 기존 데이터에는 실외 이미지가 10만 장 이상 있었으나, 학습시간 절약 및 실내 이미지 인식을 우선순위로 판단하여 실내 이미지(train/val total 약 2만 장)만 사용하였습니다.

## 프로젝트 개요

- 데이터 확인
- 데이터 구축 업체에서 생성한 모델 확인 ([https://github.com/zlstl1/yolov5_custom](https://github.com/zlstl1/yolov5_custom))
- 데이터셋 구조 변경 (Labelme에서 생성된 json 구조에서 YOLO 학습에 필요한 txt 파일로 변형)
- YOLOv8 모델을 토대로 실내 이미지 학습
- 학습된 모델에 deep sort 알고리즘을 적용하여 ID Tracking을 통한 인식율 개선 시도

## 파일 구조

```
deep_sort/
Labelme2YOLO/
src/
```

- deep sort: deep sort 알고리즘을 적용한 오픈소스(https://github.com/AarohiSingla/Tracking-and-counting-Using-YOLOv8-and-DeepSORT) 를 가져와서 사용했습니다.
- Labelme2YOLO([https://github.com/rooneysh/Labelme2YOLO](https://github.com/rooneysh/Labelme2YOLO)) 라이브러리 커스텀해서 사용했습니다.
    - 기존 데이터셋에서 rectangle(detect)과 polygon(segmentation) 분리용
    - 짝이 맞지 않는 경우가 많아 4개의 포인트만 추출했습니다.
- src
    - train_and_val.py: YOLOv8 학습 및 평가
    - predict_pic.py: 학습한 모델로 사진의 객체 인식
    - predict_cam.py: 학습한 모델로 웹캠의 객체 인식
    - predict_video.py: 학습한 모델로 영상의 객체 인식
    - predict_with_deep_sort.py: 학습한 모델에 deep sort 알고리즘 적용하여 ID Tracking
    - cnn_features: CNN 모델 특징 추출(타 모델과 비교를 위함)
    - adam_plot.ipynb: 중간에 학습을 종료한 모델의 그래프 작성
    - model_compare.ipynb: 4가지 모델의 mAP50 지표 비교 그래프

## 발표자료
- https://docs.google.com/presentation/d/19TH--L5oi7ULr-1WiK_jfP6mCm59jqohooBGfYD91vs/edit?usp=sharing

## License
Copyright 2023 Yun Oh, Taesang Cho, Hongki Cho

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
