import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 그리드 서치는 후보 값을 전부 나열하고 하나도 빠짐없이 교차 검증한다.
# 탐색이 끝나면 최적 조합으로 훈련 세트 전체를 다시 학습해 best_estimator_ 에 담아준다.
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(tr_input, tr_target)

print(gs.best_params_)
print(gs.cv_results_['mean_test_score'].round(4))

dt = gs.best_estimator_
print(dt.score(tr_input, tr_target))
print(dt.score(te_input, te_target))

# 매개변수를 여러 개 넣으면 모든 조합을 곱해서 탐색한다.
# 아래는 9 * 15 * 10 = 1350 조합이고, 5-폴드 교차 검증이므로 6750번 학습한다.
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(tr_input, tr_target)

print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']).round(4))

dt = gs.best_estimator_
print(dt.score(tr_input, tr_target))
print(dt.score(te_input, te_target))
