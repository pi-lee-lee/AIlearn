# 심층 신경망 - PyTorch 버전. deep_ke/test11.py 와 같은 결과를 내야 한다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from torch import nn
import keras   # 데이터셋 다운로드 용도로만 사용

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



criterion = nn.CrossEntropyLoss()

for opt_name in ['sgd', 'adam']:
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        print('파라미터 개수:', sum(p.numel() for p in model.parameters()))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'--- optimizer: {opt_name} ---')
    for epoch in range(5):
        model.train()
        perm = torch.randperm(len(sub_x))
        total_loss, correct = 0.0, 0

        for i in range(0, len(sub_x), 32):
            idx = perm[i:i+32]
            xb, yb = sub_x[idx], sub_y[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (pred.argmax(1) == yb).sum().item()

        print(f'epoch {epoch+1} - loss: {total_loss/len(sub_x):.4f} - accuracy: {correct/len(sub_x):.4f}')

    model.eval()
    with torch.no_grad():
        pred = model(val_x)
        loss = criterion(pred, val_y)
        acc = (pred.argmax(1) == val_y).float().mean()

    print(f'{opt_name} 검증 손실:', round(float(loss), 4))
    print(f'{opt_name} 검증 정확도:', round(float(acc), 4))
