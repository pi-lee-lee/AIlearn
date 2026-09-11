import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from torch import nn
import keras   # 데이터셋 다운로드 용도로만 사용

(tr_input, tr_target), (te_input, te_target) = keras.datasets.fashion_mnist.load_data()

torch.manual_seed(42)
np.random.seed(42)

print(tr_target[:4])
print(tr_target.shape)

tr_scaled = tr_input.reshape(-1,28*28) /255.0 # 28x28 크기의 이미지 비트맵을 1차원으로 변환 
te_scaled = te_input.reshape(-1,28*28) /255.0 # (60000, 28, 28) -> (60000, 784) 형태로 변환 

from sklearn.model_selection import train_test_split

sub_tr_input, sub_te_input, sub_tr_target, sub_te_target = train_test_split(tr_scaled, tr_target, random_state=42, test_size=0.2)


sub_x = torch.tensor(sub_tr_input, dtype=torch.float32)
sub_y = torch.tensor(sub_tr_target, dtype=torch.long)

val_x = torch.tensor(sub_te_input, dtype=torch.float32)
val_y = torch.tensor(sub_te_target, dtype=torch.long)

model = nn.Linear(28*28, 10)
loss_fn = nn.CrossEntropyLoss()
opti = torch.optim.SGD(model.parameter(), lr=0.01)


for epoch in range(5):
    model.train() # 파이토치 학습 구간 명시 
    perm = torch.randperm(len(sub_x)) #
    total_loss = 0.0
    correct = 0

    for i in range(0, len(sub_x), 32): 
        idx = perm[i:i+32]
        xb = sub_x[idx]
        yb = sub_y[idx]

        pred = model(xb)           #돌리면 10개 클래스에 대한 연산 결과가 pred에 담김 
        loss = loss_fn(pred,yb)  #yb의 정답인덱스에 해당하는 클래스의 값으로 손실 계산 진행됨 
        print('loss = ', loss.item())
        opti.zero_grad()           #이건 기울기의 초기화 누적되면 안됨 초기화 하지 않으면 당연히 opti 객체내에 기존값으로 연산시도됨 
        loss.backward()            #일단 역전파 
        opti.step()                #가중치 갱신 

        total_loss += loss.item() * len(xb) # loss.item() 은 32장 이미지에 대한 평균 손실값이다. 그래서 다시 이미지 장수 만큼 곱해준다. 원복해야 나중에 전체 이미지로 나누었을때 정확한 값이다.(어차피 평균이니 곱해도 상관없다. 순수 출력영역의 /len(sub_x) 처리의 간소화를 위해서다.)
        correct += (pred.argmax(1) == yb).sum().item() # pred.argmax(1) << 축을 0,1중 1번축 선택 그리고 거기서 가장 높은값 = 연산결과에 의해 선택된 클래스가 정답과 동일한 것들의 합(sum) 이것도 이미지 32장이다. 그래서 sum()이 있다. 32장중 모델연산처리한 결과가 정답과 동일한 것을 기록하는것이다. 

model.eval() #모델 평가모드 전환 evaluate 케라스에서는 함수로 동작시키지만 파이토치에서는 모델에 설정을 해두고 동작시킨다. 

with torch.no_grad():            #기울기 만들기 작업 없음 설정 
    pred = model(val_x)
    loss = loss_fn(pred,val_y)
    acc = (pred.argmax(1) == val_y).float().mean()

print('검증 손실:', round(float(loss), 4))
print('검증 정확도:', round(float(acc), 4))
