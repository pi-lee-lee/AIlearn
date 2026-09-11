import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(tr_input.shape, te_input.shape)

# 엑스트라 트리는 부트스트랩 샘플을 쓰지 않고 전체 훈련 세트를 쓴다.
# 대신 노드를 나눌 때 최선의 분할점을 찾지 않고 무작위로 정한다.
# 트리 하나하나는 더 엉성해지지만 그만큼 과대적합이 억제되고 학습이 빠르다.
# bootstrap=False 가 기본이라 oob_score_ 는 쓸 수 없다.

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)

scores = cross_validate(et, tr_input, tr_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(tr_input, tr_target)
print(et.score(tr_input, tr_target))
print(et.score(te_input, te_target))

# 랜덤 포레스트와 특성 중요도 비교 (test3.py 와 같은 설정)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf.fit(tr_input, tr_target)

print(data.columns.tolist())
print(et.feature_importances_.round(3))
print(rf.feature_importances_.round(3))

# 분할점을 탐색하지 않는 만큼 실제로 더 빠른지 확인
import time

start = time.time()
ExtraTreesClassifier(n_jobs=-1, random_state=42).fit(tr_input, tr_target)
print(round(time.time() - start, 3))

start = time.time()
RandomForestClassifier(n_jobs=-1, random_state=42).fit(tr_input, tr_target)
print(round(time.time() - start, 3))
