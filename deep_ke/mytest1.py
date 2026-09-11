import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import keras   # 데이터셋 다운로드 용도로만 사용

keras.utils.set_random_seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()

tr_scaled = (tr_input/255.0).reshape(-1,28*28)
te_scaled = (te_input/255.0).reshape(-1,28*28)

from sklearn.model_selection import train_test_split

sub_tr_scaled, sub_te_scaled, sub_tr_target, sub_te_target = train_test_split(tr_scaled, tr_target, test_size=0.2, random_state=42)

model = keras.Sequential([keras.layers.Input(shape=(28*28,)), keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #accuracy : 정확도 , metrics : 지표

model.fit(sub_tr_scaled, sub_tr_target, epochs=5, batch_size=32, verbose=2)

loss, acc = model.evaluate(sub_te_scaled, sub_te_target)
loss2, acc2 = model.evaluate(te_scaled, te_target)

print('손실', round(float(loss),4))
print('정확도', round(float(acc),4))

print('손실2', round(float(loss2),4))
print('정확도2', round(float(acc2),4))

# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import cross_validate

# sc = SGDClassifier(loss = 'log_loss', max_iter=5, random_state=42)

# score = cross_validate(sc, tr_scaled, tr_target, n_jobs= -1)

# print(np.mean(score['test_score']))


