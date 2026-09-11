# 로지스틱 회귀 학습 전용. 예측 과정을 라이브러리 없이 손으로 재현해서
# sklearn 결과와 한 자리까지 맞춰본다. 딥러닝의 뉴런 하나가 이것과 같은 구조다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras   # 데이터셋 다운로드 용도로만 사용

(tr_input, tr_target), _ = keras.datasets.fashion_mnist.load_data()
tr_scaled = tr_input.reshape(-1, 28*28) / 255.0

# ---------------------------------------------------------------
# 1부. 이진 분류 - 티셔츠(0) 와 바지(1) 만 남긴다
# ---------------------------------------------------------------
print('=' * 60)
print('1부. 이진 분류 (sigmoid)')
print('=' * 60)

mask = (tr_target == 0) | (tr_target == 1)
bin_x = tr_scaled[mask]
bin_y = tr_target[mask]
print('bin_x.shape:', bin_x.shape, '| 클래스:', np.unique(bin_y))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(bin_x, bin_y)

# 클래스가 2개면 가중치 세트도 1개뿐이다. z 하나로 둘을 가른다.
print('coef_.shape:', lr.coef_.shape, '| intercept_.shape:', lr.intercept_.shape)
print('파라미터 개수:', lr.coef_.size + lr.intercept_.size, '= 784 + 1')

# 확신이 강한 샘플은 확률이 0 이나 1 로 뭉개져 sigmoid 가 하는 일이 안 보인다.
# z 가 0 에 가까운(모델이 헷갈려하는) 샘플을 골라야 중간값이 드러난다.
all_z = bin_x @ lr.coef_[0] + lr.intercept_[0]
i = int(np.abs(all_z).argmin())
print('고른 샘플 번호:', i, '(z 가 0 에 가장 가까운 것)')
x = bin_x[i]
z = np.dot(lr.coef_[0], x) + lr.intercept_[0]
print('직접 계산한 z:', round(float(z), 4))
print('sklearn decision_function:', lr.decision_function([x])[0].round(4))

# sigmoid 로 확률 변환. 이 값은 "클래스 1(바지)일 확률" 이다.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

p = sigmoid(z)
print('직접 계산한 sigmoid(z):', round(float(p), 4))
print('sklearn predict_proba:', lr.predict_proba([x])[0].round(4), '(티셔츠, 바지 순)')
print('직접 판정:', '바지' if p > 0.5 else '티셔츠', '| 실제 정답:', '바지' if bin_y[i] == 1 else '티셔츠')

# ---------------------------------------------------------------
# 2부. 다중 분류 - 10개 클래스 전부
# ---------------------------------------------------------------
print()
print('=' * 60)
print('2부. 다중 분류 (softmax)')
print('=' * 60)

lr10 = LogisticRegression(max_iter=1000)
lr10.fit(tr_scaled[:10000], tr_target[:10000])

# 클래스가 10개면 가중치 세트도 10개. 같은 픽셀을 클래스마다 다르게 본다.
print('coef_.shape:', lr10.coef_.shape, '| intercept_.shape:', lr10.intercept_.shape)
print('파라미터 개수:', lr10.coef_.size + lr10.intercept_.size, '= 784*10 + 10')
print('픽셀 350 에 대한 클래스별 가중치:', lr10.coef_[:, 350].round(2))

# 여기서도 헷갈려하는 샘플을 고른다. 1등 확률이 가장 낮은 샘플.
probs = lr10.predict_proba(tr_scaled[:2000])
j = int(probs.max(axis=1).argmin())
print('고른 샘플 번호:', j, '| 1등 확률:', round(float(probs[j].max()), 4))
x = tr_scaled[j]
z10 = lr10.coef_ @ x + lr10.intercept_
print('직접 계산한 z 10개:', z10.round(2))
print('sklearn decision_function:', lr10.decision_function([x])[0].round(2))

# softmax 로 확률 변환. 지수를 취해 전부 양수로 만든 뒤 합이 1이 되게 나눈다.
def softmax(z):
    e = np.exp(z - z.max())   # 최댓값을 빼도 결과는 같다. 오버플로 방지용
    return e / e.sum()

p10 = softmax(z10)
print('직접 계산한 softmax:', p10.round(4))
print('sklearn predict_proba:', lr10.predict_proba([x])[0].round(4))
print('확률 합:', round(float(p10.sum()), 6))
print('직접 판정:', int(p10.argmax()), '| sklearn predict:', int(lr10.predict([x])[0]),
      '| 실제 정답:', int(tr_target[j]))

# ---------------------------------------------------------------
# 3부. 가중치는 무엇을 학습했는가
# ---------------------------------------------------------------
print()
print('=' * 60)
print('3부. 가중치 해석')
print('=' * 60)

# 가중치를 28x28 로 되돌리면 각 클래스가 어느 위치를 보는지 드러난다.
# 빨강(양수) = 그 픽셀이 밝으면 이 클래스에 유리, 파랑(음수) = 불리
names = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']
for i in [0, 1]:
    w = lr10.coef_[i]
    print(f'{names[i]:5s} 가중치 - 최대: {w.max():.2f} 최소: {w.min():.2f} 절편: {lr10.intercept_[i]:.2f}')

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(lr10.coef_[i].reshape(28, 28), cmap='bwr',
              vmin=-abs(lr10.coef_).max(), vmax=abs(lr10.coef_).max())
    ax.set_title(names[i])
    ax.axis('off')
fig.suptitle('class weights (red=positive, blue=negative)')

plt.show()
