import pandas as pd
import tensorflow as tf
import numpy as np



### 데이터 전처리 
data = pd.read_csv("./gpascore.csv")

# data.isnull().sum() # 데이터의 빈 값의 개수를 세어줌(각 열당 개수)

data = data.dropna() # 데이터의 빈 값이 있는 행을 삭제해줌
# data.fillna(100) 빈 값을 100으로 채워줌


### 딥러닝 모델 만들기
model = tf.keras.models.Sequential([
    # 레이어, 
    # 레이어, 
    # 레이어,
    tf.keras.layers.Dense(64, activation="tanh"), # 레이어(열) 만들기. 괄호 안 지정할 노드개수는 맘대로지만 대부분 2^n으로 사용함. 활성함수도 지정하기
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"), # 마지막 레이어(출력 담당)는 출력 개수에 따라 노드 개수가 정해진다. 출력은 확률이므로 0~1 범위지정을 위해 sigmoid.
]) # 딥러닝 모델 만들기

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']) # optimizer, loss function 지정하기
#adam 제일 많이 씀 그냥 써 손실함수는 확률 구할 때 이진교차엔트로피 사용 metrics는 다중 출력일 경우 따로 공부할거임


# [ [380, 3.21, 3], [660, 3.67, 3], [], [] ... ] 합격 여부인 admit (0열)을 제외한 나머지 값들인 학습 데이터이다.
# [ [0], [1], [1], [1], ... ] 합격 여부인 admit 열 (0열)이다. 이것이 실제값이다.


### 학습 데이터와 정답값 전처리하기
yData = data['admit'].values # .values 안 쓰면 데이터프레임 형식으로 나온다. .values 쓰면 값만 리스트로 반환됨
xData = []

for idx, rows in data.iterrows():  # iterrow 성능 제일 구리니 나중엔 찾아서 loc등 다른 방법을 잘 쓰자
  #idx는 행 번호, data.iterrow()에는 각 행의 정보가 담김. 반복할 변수 하나만 쓰면 index가 됨
  xData.append( [rows['gre'], rows['gpa'], rows['rank']])

### 학습시키기

model.fit(np.array(xData), np.array(yData), epochs=1000) #models.fit(학습데이터, 실제값, 반복횟수). *학습데이터는 numpy array 또는 tensor가 들어가야 한다* 정확도 낮으면 epochs 늘리자

  
#출력에서 loss가 nan이면 데이터에 NaN이나 inf가 들어있지 않은지 확인하자


# 예측하기
predicted = model.predict( [[750, 3.70, 3], [400, 2.2, 1]] )
print(predicted)