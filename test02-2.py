import numpy as np
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split

tr_input, te_input, tr_target, te_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 랜덤 서치는 후보를 나열하는 대신 확률 분포를 건네고 거기서 n_iter 번 뽑아 쓴다.
# uniform 은 실수, randint 는 정수를 주어진 범위에서 균등하게 뽑는다.
from scipy.stats import uniform, randint

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}

# 그리드 서치였다면 이 범위의 조합 수가 수십만 개인데, 랜덤 서치는 100번만 시도한다.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter=100, n_jobs=-1, random_state=42)
rs.fit(tr_input, tr_target)

print(rs.best_params_)
print(np.max(rs.cv_results_['mean_test_score']).round(4))

dt = rs.best_estimator_
print(dt.score(tr_input, tr_target))
print(dt.score(te_input, te_target))
