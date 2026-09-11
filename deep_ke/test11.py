# 심층 신경망 - Keras 버전. deep_to/test11.py 와 같은 결과를 내야 한다.
# test10.py 는 층이 하나뿐이라 로지스틱 회귀와 같았다. 여기서 은닉층을 끼워 넣는다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras

keras.utils.set_random_seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()
tr_scaled = tr_input.reshape(-1, 28*28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)


def build():
    # 은닉층 100개 + 출력층 10개. 은닉층 활성화가 relu 인 것이 핵심이다.
    # relu 같은 비선형 함수가 없으면 층을 쌓아도 w2(w1x) = (w2w1)x 로 접혀
    # 결국 층 하나짜리와 같아진다. 비선형이 있어야 층 쌓기가 의미를 갖는다.
    return keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


for opt_name in ['sgd', 'adam']:
    keras.utils.set_random_seed(42)
    model = build()

    if opt_name == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=0.01)
    else:
        # adam 은 파라미터마다 학습률을 자동 조절한다. sgd 는 전부 같은 폭으로 움직인다.
        opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if opt_name == 'sgd':
        model.summary()
        print('파라미터 개수: 784*100+100 + 100*10+10 =', 784*100+100 + 100*10+10)

    print(f'--- optimizer: {opt_name} ---')
    model.fit(sub_scaled, sub_target, epochs=5, batch_size=32, verbose=2)

    loss, acc = model.evaluate(val_scaled, val_target, verbose=0)
    print(f'{opt_name} 검증 손실:', round(float(loss), 4))
    print(f'{opt_name} 검증 정확도:', round(float(acc), 4))
