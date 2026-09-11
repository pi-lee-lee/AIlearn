# 합성곱 신경망 - PyTorch 버전. deep_ke/test13.py 와 같은 결과를 내야 한다.
# Keras 의 Conv2D / MaxPooling2D / Flatten 을 그대로 옮긴 것이다.
import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from torch import nn
import keras   # 데이터셋 다운로드 용도로만 사용

torch.manual_seed(42)
np.random.seed(42)

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()
# 펼치지 않는다. 채널 축을 붙이는 것은 같은데 PyTorch 는 채널이 앞이라 (샘플, 채널, 높이, 너비) 다.
tr_scaled = tr_input.reshape(-1, 1, 28, 28) / 255.0
print('tr_scaled.shape:', tr_scaled.shape)

from sklearn.model_selection import train_test_split

sub_scaled, val_scaled, sub_target, val_target = train_test_split(
    tr_scaled, tr_target, test_size=0.2, random_state=42)
print('sub_scaled.shape:', sub_scaled.shape, 'val_scaled.shape:', val_scaled.shape)

sub_x = torch.tensor(sub_scaled, dtype=torch.float32)
sub_y = torch.tensor(sub_target, dtype=torch.long)
val_x = torch.tensor(val_scaled, dtype=torch.float32)
val_y = torch.tensor(val_target, dtype=torch.long)

# 3x3 커널을 그냥 훑으면 가장자리에서는 창이 밖으로 나가 채울 수 없다. 그래서 28 이 26 으로
# 줄어든다(valid). 둘레에 0 을 한 겹 둘러 주면(same) 28 이 그대로 유지된다. 스트라이드는
# 창을 몇 칸씩 옮기는지다. 2 면 건너뛰며 훑으니 결과가 절반이 된다.
# PyTorch 는 padding 을 칸 수로 받는다. valid 는 0, same 은 (커널-1)/2 = 1 이다.
# 문자열 'same' 도 받지만 스트라이드가 1 일 때만 되므로 여기서는 숫자로 쓴다.
dummy = torch.zeros(1, 1, 28, 28)
for pad, padding, stride in [('valid', 0, 1), ('same', 1, 1), ('same', 1, 2)]:
    conv = nn.Conv2d(1, 32, kernel_size=3, padding=padding, stride=stride)
    shape = conv(dummy).shape
    print(f'padding={pad}, strides={stride} 출력 (높이, 너비, 채널):',
          (shape[2], shape[3], shape[1]))

torch.manual_seed(42)
model = nn.Sequential(
    # 필터 32개 = 서로 다른 3x3 패턴 32가지를 동시에 찾는다. 결과가 특성 맵 32장이다.
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    # 풀링은 2x2 마다 최댓값만 남긴다. 학습할 가중치가 없고 크기만 절반으로 줄인다.
    # 최댓값을 고르므로 그 패턴이 정확히 어느 칸에 있었는지는 흐려진다 - 위치가 조금
    # 밀려도 같은 답이 나오게 만드는 장치다.
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Keras 는 (7, 7, 64) 를 채널이 마지막인 순서로 펼치고 여기는 채널이 먼저인 순서로
    # 펼친다. 3136개라는 개수는 같지만 늘어선 차례가 달라서 두 프레임워크의 밀집층
    # 가중치는 서로 대응하지 않는다. 학습으로 각자 맞춰지므로 결과에는 영향이 없다.
    nn.Flatten(),
    nn.Linear(64*7*7, 100),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(100, 10)
)

LABELS = ['합성곱1', '풀링1', '합성곱2', '풀링2', '펼치기', '밀집1', '드롭아웃', '출력']

print('--- 층별 출력 크기 ---')
labels = iter(LABELS)
out = dummy
for layer in model:
    out = layer(out)
    # Keras 에서 relu 는 Conv2D/Dense 의 activation 인자라 층으로 세지 않는다. 줄을 맞춘다.
    if isinstance(layer, nn.ReLU):
        continue
    shape = tuple(out.shape[2:]) + (out.shape[1],) if out.dim() == 4 else tuple(out.shape[1:])
    params = sum(p.numel() for p in layer.parameters())
    print(f'{next(labels):<6} 출력: {str(shape):<16} 파라미터: {params}')

# 합성곱 층은 커널 한 벌만 학습해서 이미지 전체에 돌려쓴다. 그래서 입력이 몇 픽셀이든
# 파라미터 개수가 커널 크기와 필터 수에만 달려 있다 - 3*3*1*32+32 = 320 이 전부다.
# 다만 이 모델의 파라미터는 펼친 뒤의 밀집층이 거의 다 차지한다. "합성곱은 파라미터가
# 적다"는 말은 합성곱 층에 한정된 이야기이고, 모델 전체로는 test11 보다 오히려 많다.
print('합성곱 층 2개 파라미터 합:', 3*3*1*32+32 + 3*3*32*64+64)
print('밀집층 2개 파라미터 합:', 3136*100+100 + 100*10+10)
print('전체 파라미터:', sum(p.numel() for p in model.parameters()))
print('test11 밀집층만 쓴 모델의 파라미터:', 784*100+100 + 100*10+10)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def evaluate(x, y):
    # 12000장을 한 번에 통과시키면 첫 합성곱의 특성 맵만 1GB 가 넘는다. 나눠서 계산한다.
    # Keras 의 evaluate 가 내부에서 하는 일과 같다.
    model.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), 256):
            pred = model(x[i:i+256])
            total += float(criterion(pred, y[i:i+256])) * len(pred)
            correct += int((pred.argmax(1) == y[i:i+256]).sum())
    return total / len(x), correct / len(x)

# test12 와 같은 조기 종료 조건을 그대로 쓴다. 달라진 것은 모델 구조뿐이다.
PATIENCE = 2
best_loss, best_epoch, best_state, wait = float('inf'), 0, None, 0

print('--- 훈련 ---')
for epoch in range(20):
    torch.manual_seed(100 + epoch)
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

    vl, _ = evaluate(val_x, val_y)
    print(f'epoch {epoch+1:>2} - loss: {total/len(sub_x):.4f} - val_loss: {vl:.4f}')

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

loss, acc = evaluate(val_x, val_y)

print('검증 손실:', round(loss, 4))
print('검증 정확도:', round(acc, 4))
