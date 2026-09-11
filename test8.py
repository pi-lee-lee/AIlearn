import numpy as np
import urllib.request, io

with urllib.request.urlopen('https://bit.ly/fruits_300_data') as f:
    fruits = np.load(io.BytesIO(f.read()))

# test7.py 에서는 평균 이미지를 만들려고 "어느 사진이 사과인지" 를 먼저 알아야 했다.
# k-평균은 그 평균(=클러스터 중심)을 정답 없이 스스로 찾아낸다.
# 무작위 중심에서 출발해 [샘플을 가까운 중심에 배정 → 중심을 배정된 샘플의 평균으로 이동] 을
# 변화가 없을 때까지 반복한다.

fruits_2d = fruits.reshape(-1, 100*100)

print(fruits_2d.shape)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 몇 번 반복하고 멈췄는지, 각 클러스터에 몇 장씩 담겼는지
print(km.n_iter_)
print(np.unique(km.labels_, return_counts=True))

# 정답은 학습에 넣지 않았다. 아래는 오직 결과를 채점하기 위해서만 꺼내 쓴다.
answer = np.repeat(['apple', 'pineapple', 'banana'], 100)
for label in range(3):
    values, counts = np.unique(answer[km.labels_ == label], return_counts=True)
    print(label, dict(zip(values.tolist(), counts.tolist())))

# 클러스터 중심을 100x100 으로 되돌리면 test7.py 에서 손으로 만든 평균 이미지와 같은 그림이다.
# 다른 점은 이번엔 정답을 한 번도 쓰지 않고 얻었다는 것.
centers = km.cluster_centers_.reshape(-1, 100, 100)
print(centers.shape)

# transform 은 샘플에서 세 중심까지의 거리를, predict 는 그중 가장 가까운 중심의 번호를 준다.
print(km.transform(fruits_2d[100:101]).round(1))
print(km.predict(fruits_2d[100:101]))

# 실전에서는 클러스터가 몇 개인지 모른다. 이너셔(샘플과 소속 중심 사이 거리 제곱합)는
# k 를 늘리면 무조건 줄어들지만, 어느 지점부터 줄어드는 폭이 꺾인다. 그 꺾이는 지점이 적절한 k.
inertia = []
for k in range(2, 10):
    km_k = KMeans(n_clusters=k, random_state=42)
    km_k.fit(fruits_2d)
    inertia.append(km_k.inertia_)

print([round(i) for i in inertia])
print([round((inertia[i] - inertia[i+1]) / 1e8, 1) for i in range(len(inertia) - 1)])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
for ax, img, title in zip(axs, centers, ['cluster 0', 'cluster 1', 'cluster 2']):
    ax.imshow(img, cmap='gray_r')
    ax.set_title(title)
    ax.axis('off')

axs[3].plot(range(2,10), inertia, marker='o')
axs[3].set_xlabel('k')
axs[3].set_ylabel('inertia')

plt.show()
