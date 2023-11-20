import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# VGG16 모델 불러오기 (pre-trained weights 사용)
base_model = VGG16(weights='imagenet', include_top=False)

# 특정 이미지 로드 및 전처리
img_path = '이미지_파일_경로'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

layer_outputs = [layer.output for layer in base_model.layers[1:]]  # 모든 레이어의 출력 얻기 (입력 레이어는 제외)
activation_model = Model(inputs=base_model.input, outputs=layer_outputs)

activations = activation_model.predict(x)  # 각 레이어에서의 특징 맵 추출

for layer_activation in activations:
    if len(layer_activation.shape) == 4:  # 4차원 텐서인 경우 (batch_size, height, width, channels)
        n_features = min(8, layer_activation.shape[-1])  # 각 레이어의 특징 맵 개수 (최대 8개)
        size = layer_activation.shape[1]  # 특징 맵의 크기
        
        n_cols = 4  # 각 행에 표시할 특징 맵 개수
        n_rows = (n_features + n_cols - 1) // n_cols  # 특징 맵 행 개수
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(layer_activation[0, :, :, i], cmap='viridis')  # 특정 행렬의 i번째 채널을 시각화
            plt.axis('off')
        
        plt.show()