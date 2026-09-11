# 인공 신경망 - Keras 버전. deep_to/test10.py 와 같은 결과를 내야 한다.
# TensorFlow 는 Python 3.14 빌드가 없으므로 keras import 전에 백엔드를 torch 로 지정한다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras

keras.utils.set_random_seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()

print('tr_input.shape:', tr_input.shape, 'te_input.shape:', te_input.shape)

# 0~255 를 0~1 로 낮춘다. 입력이 크면 가중치 갱신 폭이 널뛰어 학습이 불안정하다.
# 28x28 이미지를 784 개 특성으로 펼친다. test7.py 에서 100x100 을 10000 으로 펼친 것과 같다.
tr_scaled = tr_input.reshape(-1, 28*28) / 255.0
te_scaled = te_input.reshape(-1, 28*28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

# 밀집층 하나뿐인 신경망. 784 개 입력이 10 개 출력에 전부 연결된다.
# 활성화 함수가 softmax 이고 출력이 10개이므로 사실상 다중 분류 로지스틱 회귀와 같은 구조다.
# 차이는 경사 하강법을 미니배치로 반복 수행한다는 점.
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# sparse_categorical_crossentropy: 타깃이 원-핫이 아니라 정수(0~9)일 때 쓰는 손실.
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 에포크당 6.6초. deep_to/test10.py 의 같은 학습이 0.4초이므로 16배 느리다.
# 원인은 배치당 약 4ms 의 Keras 오버헤드(numpy->텐서 변환, 메트릭 누적, 콜백 처리)이고
# 배치가 1500개라 6초가 된다. 784x10 짜리 행렬 연산 자체는 순식간에 끝난다.
# 모델이 작을수록 오버헤드 비중이 커지므로 지금이 최악의 조건이고, 모델이 커지면 격차는 줄어든다.
model.fit(sub_scaled, sub_target, epochs=5, batch_size=32, verbose=2)

loss, acc = model.evaluate(val_scaled, val_target, verbose=0)
print('검증 손실:', round(float(loss), 4))
print('검증 정확도:', round(float(acc), 4))
