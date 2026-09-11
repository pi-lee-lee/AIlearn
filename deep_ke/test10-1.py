# 인공 신경망과 비교할 기준선. 신경망을 쓰기 전에 로지스틱 회귀로 같은 문제를 풀어본다.
# 프레임워크와 무관한 sklearn 코드라 deep_to 에 짝을 두지 않는다(deep_ke 만 존재).
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras   # 데이터셋 다운로드 용도로만 사용

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()
print('tr_input.shape:', tr_input.shape, 'te_input.shape:', te_input.shape)

tr_scaled = tr_input.reshape(-1, 28*28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

# 확률적 경사 하강법 + 로지스틱 손실. Ailearn.py 에서 생선 데이터로 쓴 것과 같은 모델이다.
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)

scores = cross_validate(sc, tr_scaled, tr_target, n_jobs=-1)
print('교차검증 점수:', scores['test_score'].round(4))
print('교차검증 평균:', np.mean(scores['test_score']).round(4))
print('폴드당 학습 초:', np.mean(scores['fit_time']).round(2))

# test10.py 와 같은 분할로 다시 재서 신경망과 직접 비교한다
from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
sc.fit(sub_scaled, sub_target)
print('훈련 정확도:', round(sc.score(sub_scaled, sub_target), 4))
print('검증 정확도:', round(sc.score(val_scaled, val_target), 4))

# 신경망(test10.py)의 밀집층 하나짜리 구조는 사실 이 모델과 수학적으로 같다.
# 파라미터도 784*10+10 = 7850 개로 동일하다.
print('파라미터 개수:', sc.coef_.size + sc.intercept_.size)
