import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 테스트 세트는 마지막에 딱 한 번만 써야 한다. 모델을 고르는 동안 참고할
# 점수가 필요하니 훈련 세트를 한 번 더 쪼개서 검증 세트를 만든다.
sub_input, val_input, sub_target, val_target = train_test_split(tr_input, tr_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# 검증 세트를 한 번만 떼면 어느 20%가 뽑혔느냐에 따라 점수가 출렁인다.
# 교차 검증은 훈련 세트를 k조각으로 나눠 번갈아 검증 세트로 쓰고 평균을 낸다.
from sklearn.model_selection import cross_validate

dt = DecisionTreeClassifier(random_state=42)
scores = cross_validate(dt, tr_input, tr_target)
print(scores['test_score'])
print(np.mean(scores['test_score']))

# 분류에서는 폴드마다 클래스 비율을 유지하는 StratifiedKFold 가 기본으로 쓰인다.
# 직접 넘기면 폴드 수를 바꾸거나 shuffle 을 켤 수 있다.
from sklearn.model_selection import StratifiedKFold

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, tr_input, tr_target, cv=splitter)
print(np.mean(scores['test_score']))
