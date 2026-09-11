import numpy as np
import urllib.request, io

with urllib.request.urlopen('https://bit.ly/fruits_300_data') as f:
    fruits = np.load(io.BytesIO(f.read()))

fruits_2d = fruits.reshape(-1, 100*100)
target = np.repeat([0, 1, 2], 100)

# PCA 도 스케일러와 같다. 훈련 세트로만 fit 하고 테스트 세트에는 transform 만 적용한다.
# 테스트 세트까지 넣어 fit 하면 평가에 쓸 데이터를 축을 정하는 데 미리 참고한 셈이 된다.
from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(
    fruits_2d, target, test_size=0.2, stratify=target, random_state=42)

print('tr_input.shape:', tr_input.shape, 'te_input.shape:', te_input.shape)

# 원본 10000개 특성으로 기준을 잡는다
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(tr_input, tr_target)
print('원본 훈련 점수:', round(lr.score(tr_input, tr_target), 4))
print('원본 테스트 점수:', round(lr.score(te_input, te_target), 4))

# 차원 축소. 데이터가 가장 넓게 퍼진 방향(주성분)을 훈련 세트에서만 찾는다.
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(tr_input)

print('pca.components_.shape:', pca.components_.shape)
print('explained_variance_ratio_[:5]:', pca.explained_variance_ratio_[:5].round(3))
print('explained_variance_ratio_ 합:', pca.explained_variance_ratio_.sum().round(4))

# 훈련 세트에서 찾은 축을 양쪽에 똑같이 적용한다
tr_pca = pca.transform(tr_input)
te_pca = pca.transform(te_input)
print('tr_pca.shape:', tr_pca.shape, 'te_pca.shape:', te_pca.shape)

lr.fit(tr_pca, tr_target)
print('PCA50 훈련 점수:', round(lr.score(tr_pca, tr_target), 4))
print('PCA50 테스트 점수:', round(lr.score(te_pca, te_target), 4))

# inverse_transform 으로 되돌리면 무엇을 잃었는지 알 수 있다.
# 50개로 줄였다 복원했으니 원본과 같을 수 없고, 그 차이가 버린 분산에 해당한다.
tr_inverse = pca.inverse_transform(tr_pca)
print('tr_inverse.shape:', tr_inverse.shape)
print('복원 평균 오차(0~255):', np.abs(tr_input - tr_inverse).mean().round(2))

# n_components 에 0~1 실수를 주면 그 비율의 분산을 담는 최소 개수를 알아서 정한다.
pca = PCA(n_components=0.5)
pca.fit(tr_input)
print('pca.n_components_:', pca.n_components_)

tr_pca = pca.transform(tr_input)
te_pca = pca.transform(te_input)

lr.fit(tr_pca, tr_target)
print('PCA50% 훈련 점수:', round(lr.score(tr_pca, tr_target), 4))
print('PCA50% 테스트 점수:', round(lr.score(te_pca, te_target), 4))

# 군집은 정답을 쓰지 않으므로 훈련 세트 안에서 그대로 확인한다
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(tr_pca)

answer = np.array(['apple', 'pineapple', 'banana'])[tr_target]
for label in range(3):
    values, counts = np.unique(answer[km.labels_ == label], return_counts=True)
    print('클러스터', label, ':', dict(zip(values.tolist(), counts.tolist())))

# 2개로 줄이면 산점도로 그릴 수 있다. 10000차원에서는 불가능했던 일이다.
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(14, 4))

for i in range(3):
    axs[0].scatter(tr_pca[km.labels_ == i, 0], tr_pca[km.labels_ == i, 1])
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')

axs[1].imshow(tr_input[0].reshape(100, 100), cmap='gray_r')
axs[1].set_title('original')
axs[2].imshow(tr_inverse[0].reshape(100, 100), cmap='gray_r')
axs[2].set_title('restored (50)')
axs[3].imshow(pca.components_[0].reshape(100, 100), cmap='gray_r')
axs[3].set_title('first component')
for ax in axs[1:]:
    ax.axis('off')

plt.show()
