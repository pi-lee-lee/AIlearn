# 합성곱 신경망 - Keras 버전. deep_to/test13.py 와 같은 결과를 내야 한다.
# test10~12 는 28x28 이미지를 784개 숫자로 펼쳐서 넣었다. 펼치는 순간 어느 픽셀이
# 어느 픽셀 옆에 있었는지가 사라진다. 합성곱은 이미지를 2차원 그대로 두고 작은
# 창(커널)을 움직이며 훑으므로 그 이웃 관계가 남는다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras

keras.utils.set_random_seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()
# 펼치지 않는다. 흑백이라 채널 축이 없으므로 1 을 붙여 (샘플, 높이, 너비, 채널) 로 만든다.
tr_scaled = tr_input.reshape(-1, 28, 28, 1) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

# 3x3 커널을 그냥 훑으면 가장자리에서는 창이 밖으로 나가 채울 수 없다. 그래서 28 이 26 으로
# 줄어든다(valid). 둘레에 0 을 한 겹 둘러 주면(same) 28 이 그대로 유지된다. 스트라이드는
# 창을 몇 칸씩 옮기는지다. 2 면 건너뛰며 훑으니 결과가 절반이 된다.
dummy = np.zeros((1, 28, 28, 1), dtype='float32')
for pad, stride in [('valid', 1), ('same', 1), ('same', 2)]:
    conv = keras.layers.Conv2D(32, kernel_size=3, padding=pad, strides=stride)
    print(f'padding={pad}, strides={stride} 출력 (높이, 너비, 채널):',
          tuple(conv(dummy).shape[1:]))

keras.utils.set_random_seed(42)
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    # 필터 32개 = 서로 다른 3x3 패턴 32가지를 동시에 찾는다. 결과가 특성 맵 32장이다.
    keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
    # 풀링은 2x2 마다 최댓값만 남긴다. 학습할 가중치가 없고 크기만 절반으로 줄인다.
    # 최댓값을 고르므로 그 패턴이 정확히 어느 칸에 있었는지는 흐려진다 - 위치가 조금
    # 밀려도 같은 답이 나오게 만드는 장치다.
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax')
])

LABELS = ['합성곱1', '풀링1', '합성곱2', '풀링2', '펼치기', '밀집1', '드롭아웃', '출력']

print('--- 층별 출력 크기 ---')
out = keras.ops.convert_to_tensor(dummy)
for label, layer in zip(LABELS, model.layers):
    out = layer(out)
    params = sum(int(np.prod(w.shape)) for w in layer.weights)
    print(f'{label:<6} 출력: {str(tuple(out.shape[1:])):<16} 파라미터: {params}')

# 합성곱 층은 커널 한 벌만 학습해서 이미지 전체에 돌려쓴다. 그래서 입력이 몇 픽셀이든
# 파라미터 개수가 커널 크기와 필터 수에만 달려 있다 - 3*3*1*32+32 = 320 이 전부다.
# 다만 이 모델의 파라미터는 펼친 뒤의 밀집층이 거의 다 차지한다. "합성곱은 파라미터가
# 적다"는 말은 합성곱 층에 한정된 이야기이고, 모델 전체로는 test11 보다 오히려 많다.
print('합성곱 층 2개 파라미터 합:', 3*3*1*32+32 + 3*3*32*64+64)
print('밀집층 2개 파라미터 합:', 3136*100+100 + 100*10+10)
print('전체 파라미터:', model.count_params())
print('test11 밀집층만 쓴 모델의 파라미터:', 784*100+100 + 100*10+10)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# test12 와 같은 조기 종료 조건을 그대로 쓴다. 달라진 것은 모델 구조뿐이다.
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                   restore_best_weights=True)

print('--- 훈련 ---')
history = model.fit(sub_scaled, sub_target, epochs=20, batch_size=32,
                    validation_data=(val_scaled, val_target),
                    callbacks=[es], verbose=0)

for i in range(len(history.history['loss'])):
    print(f'epoch {i+1:>2} - loss: {history.history["loss"][i]:.4f}'
          f' - val_loss: {history.history["val_loss"][i]:.4f}')

print('실제로 돈 에포크:', len(history.history['loss']))
print('되돌린 최적 에포크:', es.best_epoch + 1)

loss, acc = model.evaluate(val_scaled, val_target, verbose=0)
print('검증 손실:', round(float(loss), 4))
print('검증 정확도:', round(float(acc), 4))
