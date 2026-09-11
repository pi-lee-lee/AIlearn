# 인공 신경망 - PyTorch 버전. deep_ke/test10.py 와 같은 결과를 내야 한다.
# Keras 가 감춰주는 학습 루프(배치 나누기, 순전파, 손실, 역전파, 갱신)를 직접 쓴다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from torch import nn
import keras   # 데이터셋 다운로드 용도로만 사용

torch.manual_seed(42)
np.random.seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()

print('tr_input.shape:', tr_input.shape, 'te_input.shape:', te_input.shape)

tr_scaled = tr_input.reshape(-1, 28*28) / 255.0
te_scaled = te_input.reshape(-1, 28*28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

# numpy 배열을 torch 텐서로. 파이토치는 float32 가 기본이다.
sub_x = torch.tensor(sub_scaled, dtype=torch.float32)
sub_y = torch.tensor(sub_target, dtype=torch.long)

val_x = torch.tensor(val_scaled, dtype=torch.float32)
val_y = torch.tensor(val_target, dtype=torch.long)

# Dense(10, softmax) 에 해당. 파이토치는 Linear 가 곱셈만 하고 활성화는 따로 붙인다.
# 여기서 softmax 를 넣지 않는 이유는 아래 CrossEntropyLoss 가 내부에 이미 갖고 있어서다.
# 두 번 적용하면 학습이 망가진다.
model = nn.Linear(784, 10)
criterion = nn.CrossEntropyLoss()          # = sparse_categorical_crossentropy + softmax
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print('파라미터 개수:', sum(p.numel() for p in model.parameters()))

# Keras 의 model.fit(epochs=5, batch_size=32) 을 직접 푼 것.
# 에포크당 0.4초로 deep_ke/test10.py 의 6.6초보다 16배 빠르다. 계산량은 같고,
# 차이는 Keras 계층이 배치마다 붙이는 부가 작업(약 4ms)에서 나온다.
for epoch in range(5):
    model.train()
    perm = torch.randperm(len(sub_x))
    total_loss = 0.0
    correct = 0

    for i in range(0, len(sub_x), 32):
        idx = perm[i:i+32]
        xb, yb = sub_x[idx], sub_y[idx]

        pred = model(xb)                  # 순전파
        loss = criterion(pred, yb)        # 손실

        optimizer.zero_grad()             # 이전 기울기 초기화
        loss.backward()                   # 역전파: 기울기 계산
        optimizer.step()                  # 가중치 갱신

        total_loss += loss.item() * len(xb)
        correct += (pred.argmax(1) == yb).sum().item()

    print(f'epoch {epoch+1} - loss: {total_loss/len(sub_x):.4f} - accuracy: {correct/len(sub_x):.4f}')

model.eval()
with torch.no_grad():                     # 평가 때는 기울기를 만들지 않는다
    pred = model(val_x)
    loss = criterion(pred, val_y)
    acc = (pred.argmax(1) == val_y).float().mean()

print('검증 손실:', round(float(loss), 4))
print('검증 정확도:', round(float(acc), 4))
