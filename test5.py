import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(tr_input.shape, te_input.shape)

# 랜덤 포레스트/엑스트라 트리는 깊은 트리를 잔뜩 만들어 평균내는(배깅) 방식이지만,
# 그래디언트 부스팅은 깊이 3짜리 얕은 트리를 순차적으로 추가하면서
# 앞선 트리가 남긴 오차를 경사 하강법으로 줄여나간다(부스팅).
# 트리를 순서대로 만들어야 하므로 n_jobs 로 병렬화할 수 없다.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

gb = GradientBoostingClassifier(random_state=42)

scores = cross_validate(gb, tr_input, tr_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 기본값(트리 100개, max_depth=3)은 오히려 과소적합에 가깝다.
# 트리를 5배로 늘리고 학습률을 2배로 올려도 부스팅은 과대적합에 강한 편이다.
gb2 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)

scores = cross_validate(gb2, tr_input, tr_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb2.fit(tr_input, tr_target)
print(gb2.score(tr_input, tr_target))
print(gb2.score(te_input, te_target))

print(data.columns.tolist())
print(gb2.feature_importances_.round(3))

# 위 두 cross_validate 는 평균만 찍었다. test02.py 에서 본 것처럼 폴드별 점수를
# 직접 펼쳐보면 이 평균이 얼마나 믿을 만한 숫자인지 알 수 있다.
from sklearn.model_selection import StratifiedKFold

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(gb2, tr_input, tr_target, return_train_score=True, cv=splitter, n_jobs=-1)

print(scores['test_score'].round(4))
print(np.mean(scores['test_score']).round(4), np.std(scores['test_score']).round(4))

# 폴드 하나당 학습 시간 비교. 위쪽 n_jobs=-1 은 폴드끼리만 병렬로 돌릴 뿐,
# 부스팅은 트리를 순서대로 만들어야 해서 모델 내부는 병렬화되지 않는다.
from sklearn.ensemble import RandomForestClassifier

rf_scores = cross_validate(RandomForestClassifier(n_jobs=-1, random_state=42),
                           tr_input, tr_target, cv=splitter, n_jobs=-1)

print(np.mean(scores['fit_time']).round(3), np.mean(rf_scores['fit_time']).round(3))
print(np.mean(rf_scores['test_score']).round(4))
