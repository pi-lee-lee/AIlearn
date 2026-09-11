import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

print(tr_input.shape, te_input.shape)

# 히스토그램 기반 부스팅은 특성값을 미리 256개 구간으로 나눠놓고 그 경계에서만
# 분할점을 찾는다. 후보가 256개로 고정되니 샘플이 아무리 늘어도 속도가 잘 버틴다.
# 구간 하나는 누락값 몫으로 빼둬서 결측치를 따로 채우지 않아도 된다.
# 트리 개수를 n_estimators 가 아니라 max_iter 로 지정하는 점도 다르다.

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

hgb = HistGradientBoostingClassifier(random_state=42)

scores = cross_validate(hgb, tr_input, tr_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

hgb.fit(tr_input, tr_target)
print(hgb.score(tr_input, tr_target))
print(hgb.score(te_input, te_target))

# 이 모델에는 feature_importances_ 가 없다. 대신 특성을 하나씩 무작위로 섞어
# 점수가 얼마나 떨어지는지 재는 순열 중요도를 쓴다. 모델 종류를 가리지 않고,
# 훈련 세트뿐 아니라 테스트 세트에도 적용할 수 있는 게 장점이다.
from sklearn.inspection import permutation_importance

print(data.columns.tolist())

result = permutation_importance(hgb, tr_input, tr_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean.round(3))

result = permutation_importance(hgb, te_input, te_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean.round(3))

# test5.py 의 그래디언트 부스팅과 폴드당 학습 시간 비교
from sklearn.ensemble import GradientBoostingClassifier

gb_scores = cross_validate(GradientBoostingClassifier(random_state=42), tr_input, tr_target, n_jobs=-1)
print(np.mean(scores['fit_time']).round(3), np.mean(gb_scores['fit_time']).round(3))
print(np.mean(gb_scores['test_score']).round(4))
