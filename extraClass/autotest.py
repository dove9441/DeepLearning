import tensorflow as tf
import matplotlib.pyplot as plt # 이미지를 파이썬에서 띄워보기 위해
import numpy as np # reshape 기능을 사용하기 휘해
import time

#data = tf.keras.datasets.fashion_mnist.load_data() # 텐서플로우 기본 제공 데이터셋(쇼핑몰)
# 안의 내용물은 다음과 같다.
# ( ( trainX, trainY), (testX, testY) ) trainY는 정답값(아래의 class_names의 인덱스)이고, 왼쪽처럼 튜플로 들어있다.

#print(data[0][0].shape) # (60000,28,28)로 나온다. 28x28 이미지 6만개이다.

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0 # 0~255 아닌 0~1 사용하면 속도나 성능 개선이 있을 수 있음. 선택사항임.

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1))
testX = testX.reshape( (testX.shape[0], 28, 28, 1))


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

### 모델 만들기

# 레이어 구성 순서 : [Conv2D -> MaxPooling2D] x N(반복하든 말든) -> Flatten -> Danse

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)), 
    tf.keras.layers.MaxPooling2D( (2,2) ), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# 텐서보드를 이용한 시각적 분석 tensorboard --logdir "로그 경로" 터미널에 입력하면 볼 수 있음
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format('test'+str(int(time.time()))))
# EalryStopping을 이용하면 epochs가 많아도 알아서 overfitting이 일어나는 지점에서 멈춰줌
es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=3, mode='min') #patience만큼 monitor가 'mode'로 최신화되지 않으면 멈춰줌 

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, callbacks=[tensorboard, es]) 




