import numpy as np
import urllib.request, io

# test9.py 는 train_test_split 으로 나눈 뒤 훈련 세트로만 PCA 를 fit 한다.
# 이 파일은 분리 없이 전체 300장으로 교차 검증하는 버전이다.
# 테스트 세트가 60장뿐이라 test9.py 는 점수가 전부 1.0 으로 뭉개지는데,
# 5-폴드 교차 검증은 300장을 번갈아 검증에 쓰므로 0.9967 로 변별이 남는다.

with urllib.request.urlopen('https://bit.ly/fruits_300_data') as f:
    fruits = np.load(io.BytesIO(f.read()))

fruits_2d = fruits.reshape(-1, 100*100)
target = np.repeat([0, 1, 2], 100)
print('fruits_2d.shape:', fruits_2d.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

print('pca.components_.shape:', pca.components_.shape)
print('explained_variance_ratio_[:5]:', pca.explained_variance_ratio_[:5].round(3))
print('explained_variance_ratio_ 합:', pca.explained_variance_ratio_.sum().round(4))

fruits_pca = pca.transform(fruits_2d)
print('fruits_pca.shape:', fruits_pca.shape)

# 되돌려서 무엇을 잃었는지 확인
fruits_inverse = pca.inverse_transform(fruits_pca)
print('fruits_inverse.shape:', fruits_inverse.shape)
print('복원 평균 오차(0~255):', np.abs(fruits_2d - fruits_inverse).mean().round(2))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

lr = LogisticRegression(max_iter=1000)

scores = cross_validate(lr, fruits_2d, target)
print('원본 10000개 - 교차검증:', np.mean(scores['test_score']).round(4),
      '폴드당 초:', np.mean(scores['fit_time']).round(3))

scores = cross_validate(lr, fruits_pca, target)
print('PCA 50개 - 교차검증:', np.mean(scores['test_score']).round(4),
      '폴드당 초:', np.mean(scores['fit_time']).round(3))

# n_components 에 0~1 실수를 주면 그 비율의 분산을 담는 최소 개수를 알아서 정한다
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print('pca.n_components_:', pca.n_components_)

fruits_pca = pca.transform(fruits_2d)

scores = cross_validate(lr, fruits_pca, target)
print('PCA 2개 - 교차검증:', np.mean(scores['test_score']).round(4),
      '폴드당 초:', np.mean(scores['fit_time']).round(3))

# 축소한 데이터로 k-평균을 돌려도 test8.py 와 같은 군집이 나오는지 확인
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)

answer = np.repeat(['apple', 'pineapple', 'banana'], 100)
for label in range(3):
    values, counts = np.unique(answer[km.labels_ == label], return_counts=True)
    print('클러스터', label, ':', dict(zip(values.tolist(), counts.tolist())))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(14, 4))

for i in range(3):
    axs[0].scatter(fruits_pca[km.labels_ == i, 0], fruits_pca[km.labels_ == i, 1])
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')

axs[1].imshow(fruits_2d[0].reshape(100, 100), cmap='gray_r')
axs[1].set_title('original')
axs[2].imshow(fruits_inverse[0].reshape(100, 100), cmap='gray_r')
axs[2].set_title('restored (50)')
axs[3].imshow(pca.components_[0].reshape(100, 100), cmap='gray_r')
axs[3].set_title('first component')
for ax in axs[1:]:
    ax.axis('off')

plt.show()
