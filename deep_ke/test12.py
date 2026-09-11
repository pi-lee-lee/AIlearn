# 신경망 모델 훈련 - Keras 버전. deep_to/test12.py 와 같은 결과를 내야 한다.
# test11.py 는 훈련 손실만 봤다. 훈련 손실은 에포크가 갈수록 계속 내려가므로
# 언제 멈춰야 하는지 알려주지 못한다. 여기서는 검증 손실을 함께 재서 그 지점을 찾는다.
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

EPOCHS = 20


def build(dropout=0.0):
    layers = [keras.layers.Input(shape=(784,)),
              keras.layers.Dense(100, activation='relu')]
    # 드롭아웃은 훈련 중에만 은닉 뉴런 일부를 무작위로 끈다. 특정 뉴런에 의존하지
    # 못하게 만들어 암기를 막는 장치다. 폭을 줄이는 것과 달리 표현력은 그대로 둔다.
    # 평가 때는 전부 켜지므로 evaluate 결과에는 무작위성이 없다.
    if dropout > 0:
        layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(10, activation='softmax'))
    return keras.Sequential(layers)


# 드롭아웃 없는 쪽과 있는 쪽을 같은 조건으로 돌려 훈련-검증 격차를 비교한다
for rate in [0.0, 0.2]:
    keras.utils.set_random_seed(42)
    model = build(rate)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f'--- dropout: {rate} ---')
    # validation_data 를 넘기면 에포크마다 검증 손실을 함께 계산해 history 에 담는다
    history = model.fit(sub_scaled, sub_target, epochs=EPOCHS, batch_size=32,
                        validation_data=(val_scaled, val_target), verbose=0)

    for i in range(EPOCHS):
        print(f'epoch {i+1:>2} - loss: {history.history["loss"][i]:.4f}'
              f' - val_loss: {history.history["val_loss"][i]:.4f}')

    best = int(np.argmin(history.history['val_loss']))
    print('최저 검증 손실 에포크:', best + 1)
    print('최저 검증 손실:', round(float(history.history['val_loss'][best]), 4))
    print('마지막 검증 손실:', round(float(history.history['val_loss'][-1]), 4))
    print('마지막 검증 정확도:', round(float(history.history['val_accuracy'][-1]), 4))

# 조기 종료. patience 만큼 검증 손실이 나아지지 않으면 멈추고 최적 가중치로 되돌린다.
# 위에서 최저 지점을 눈으로 찾았다면, 이건 그 작업을 자동으로 하는 것이다.




keras.utils.set_random_seed(42)
model = build(0.2)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                   restore_best_weights=True)

print('--- early stopping ---')
history = model.fit(sub_scaled, sub_target, epochs=EPOCHS, batch_size=32,
                    validation_data=(val_scaled, val_target),
                    callbacks=[es], verbose=0)

print('실제로 돈 에포크:', len(history.history['loss']))
print('되돌린 최적 에포크:', es.best_epoch + 1)

loss, acc = model.evaluate(val_scaled, val_target, verbose=0)
print('검증 손실:', round(float(loss), 4))
print('검증 정확도:', round(float(acc), 4))
