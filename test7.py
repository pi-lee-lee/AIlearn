import numpy as np

# 여기서부터는 비지도 학습이다. 지금까지는 target 을 주고 정답을 맞히게 시켰지만
# 군집은 정답 없이 비슷한 것끼리 묶는다. train_test_split 이 없는 이유이기도 하다.
# 데이터도 CSV 가 아니라 흑백 사진 300장을 담은 넘파이 배열이다.
import urllib.request, io

with urllib.request.urlopen('https://bit.ly/fruits_300_data') as f:
    fruits = np.load(io.BytesIO(f.read()))

print('1)))')
print(fruits.shape)

# 100x100 사진 300장. 앞 100장 사과, 다음 100장 파인애플, 마지막 100장 바나나.
# 사진은 흑백 반전되어 있어서 배경(원래 흰 종이)이 0 에 가깝고 과일이 밝다.
# 관심 대상인 과일에 큰 값을 줘야 평균을 낼 때 과일이 주도권을 갖는다.
print('2)))')
print(fruits[0, 0, :10])

# 사진 한 장을 10000개 픽셀짜리 1차원 벡터로 펼친다. 픽셀 하나가 특성 하나인 셈이다.
apple = fruits[:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:].reshape(-1, 100*100)

print('3)))')
print(apple.shape, pineapple.shape, banana.shape)

# 1) 샘플별 픽셀 평균 하나로 갈라보기 (axis=1: 사진 한 장 안의 10000개 픽셀을 평균)
print('4)))')
print(apple.mean(axis=1).mean().round(1),
      pineapple.mean(axis=1).mean().round(1),
      banana.mean(axis=1).mean().round(1))

# 바나나는 화면에서 차지하는 면적이 작아 평균이 뚝 떨어지지만 사과와 파인애플은 겹친다.
# 사과 평균보다 밝은 파인애플이 몇 장이나 되는지 세어보면 숫자로 확인된다.
print('5)))')
print((pineapple.mean(axis=1) > apple.mean(axis=1).mean()).sum())

# 2) 픽셀별 평균으로 바꿔보기 (axis=0: 사진 100장에 걸쳐 같은 위치 픽셀끼리 평균)
# 값 하나로 뭉개지 않고 10000개 위치별 밝기 패턴을 남기므로 모양 차이가 살아남는다.
apple_mean = apple.mean(axis=0).reshape(100, 100)
pineapple_mean = pineapple.mean(axis=0).reshape(100, 100)
banana_mean = banana.mean(axis=0).reshape(100, 100)

print('6)))')
print(apple_mean.shape)

# 3) 평균 이미지와 가장 가까운 사진 100장을 골라 군집으로 삼는다.
# 300장 전체에서 픽셀별 차이의 절댓값을 평균내면 사진 한 장당 거리 하나가 나온다.
for name, mean_img, answer in [('apple', apple_mean, range(0, 100)),
                               ('pineapple', pineapple_mean, range(100, 200)),
                               ('banana', banana_mean, range(200, 300))]:
    abs_diff = np.abs(fruits - mean_img)
    abs_mean = np.mean(abs_diff, axis=(1, 2))
    picked = np.argsort(abs_mean)[:100]
    hit = np.isin(picked, list(answer)).sum()
    print('7)))')
    print(name, hit)

# 사과와 파인애플이 100장을 다 맞히지 못하는 건 평균 이미지가 서로 닮았기 때문이다.
# 더 중요한 한계는 따로 있다. apple_mean 을 만들려면 어느 사진이 사과인지 이미 알아야 했다.
# 정답을 쓴 셈이니 엄밀히 말하면 아직 비지도 학습이 아니다.
# 정답 없이 이 평균(=클러스터 중심)을 스스로 찾아내는 것이 다음 단계인 k-평균이다.

import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 3, figsize=(9, 3))

# axs[0].bar(range(10000), apple.mean(axis=0))
# axs[1].bar(range(10000), pineapple.mean(axis=0))
# axs[2].bar(range(10000), banana.mean(axis=0))

# plt.show()

# for ax, img, title in zip(axs, [apple_mean, pineapple_mean, banana_mean], ['apple', 'pineapple', 'banana']):
#     ax.imshow(img, cmap='gray_r')
#     ax.set_title(title)
#     ax.axis('off')

# plt.show()

abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print('8)))')
print(abs_mean.shape)

apple_idx = np.argsort(abs_mean)[:100]
apple_idx = apple_idx.reshape(10,10)

fig, axs = plt.subplots(10,10, figsize=(10,10))

for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_idx[i,j]], cmap='gray_r')
        axs[i,j].axis('off')

plt.show()

