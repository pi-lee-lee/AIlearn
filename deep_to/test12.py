# 신경망 모델 훈련 - PyTorch 버전. deep_ke/test12.py 와 같은 결과를 내야 한다.
# Keras 의 validation_data / EarlyStopping / restore_best_weights 를 직접 푼 것이다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from torch import nn
import keras   # 데이터셋 다운로드 용도로만 사용

torch.manual_seed(42)
np.random.seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()
tr_scaled = tr_input.reshape(-1, 28*28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

sub_x = torch.tensor(sub_scaled, dtype=torch.float32)
sub_y = torch.tensor(sub_target, dtype=torch.long)
val_x = torch.tensor(val_scaled, dtype=torch.float32)
val_y = torch.tensor(val_target, dtype=torch.long)

EPOCHS = 20
criterion = nn.CrossEntropyLoss()


def build(dropout=0.0):
    layers = [nn.Linear(784, 100), nn.ReLU()]
    # Keras 의 Dropout 층과 같은 것. test10/test11 에서는 model.train()/eval() 이
    # 아무 효과가 없었는데, 드롭아웃이 들어온 지금부터 실제로 동작이 갈린다.
    # train 모드에서만 뉴런을 끄고 eval 모드에서는 전부 켠다.
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(100, 10))
    return nn.Sequential(*layers)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        return float(criterion(pred, y)), float((pred.argmax(1) == y).float().mean())


def train_epoch(model, optimizer):
    model.train()
    perm = torch.randperm(len(sub_x))
    total = 0.0
    for i in range(0, len(sub_x), 32):
        idx = perm[i:i+32]
        xb, yb = sub_x[idx], sub_y[idx]
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(sub_x)


for rate in [0.0, 0.2]:
    torch.manual_seed(42)
    model = build(rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'--- dropout: {rate} ---')
    val_losses, val_accs = [], []
    for epoch in range(EPOCHS):
        torch.manual_seed(100 + epoch)
        tr_loss = train_epoch(model, optimizer)
        vl, va = evaluate(model, val_x, val_y)
        val_losses.append(vl)
        val_accs.append(va)
        print(f'epoch {epoch+1:>2} - loss: {tr_loss:.4f} - val_loss: {vl:.4f}')

    best = int(np.argmin(val_losses))
    print('최저 검증 손실 에포크:', best + 1)
    print('최저 검증 손실:', round(val_losses[best], 4))
    print('마지막 검증 손실:', round(val_losses[-1], 4))
    print('마지막 검증 정확도:', round(val_accs[-1], 4))

# 조기 종료. Keras 의 EarlyStopping(patience=2, restore_best_weights=True) 를 손으로 쓴다.
# 최적 가중치는 state_dict 를 복사해 들고 있다가 끝에 되돌린다.
torch.manual_seed(42)
model = build(0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

PATIENCE = 2
best_loss, best_epoch, best_state, wait = float('inf'), 0, None, 0

print('--- early stopping ---')
for epoch in range(EPOCHS):
    torch.manual_seed(100 + epoch)
    train_epoch(model, optimizer)
    vl, _ = evaluate(model, val_x, val_y)

    if vl < best_loss:
        best_loss, best_epoch, wait = vl, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            break

ran = epoch + 1
model.load_state_dict(best_state)

print('실제로 돈 에포크:', ran)
print('되돌린 최적 에포크:', best_epoch + 1)

loss, acc = evaluate(model, val_x, val_y)
print('검증 손실:', round(loss, 4))
print('검증 정확도:', round(acc, 4))
