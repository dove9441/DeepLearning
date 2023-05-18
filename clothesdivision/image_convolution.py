import tensorflow as tf
import matplotlib.pyplot as plt # 이미지를 파이썬에서 띄워보기 위해
import numpy as np # reshape 기능을 사용하기 휘해

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
    #input shape 없으면 summery 말고도 ndim 오류가 나는데, 최소 4차원 데이터가 필요한데(60000,28,28,1 등) 사용 데이터는 (28,28)이기 때문에 오류가 남. 그래서 []추가해서 바꿔줘야 함. 위의 reshape 코드 사용.
    # 컬러일 경우 [255,145,158] 이런 게 1쌍이기 때문에 마지막을 3으로 해줘야 함
    # *convolution layer는 이렇게 만듬. 32는 복사본 개수, 3,3은 kernel 사이즈인데, 3x3부터 맘대로 하면 됨. relu쓰는이유는 마이너스 제거*
    tf.keras.layers.MaxPooling2D( (2,2) ), #이미지 크기를 줄이고 가운데로 모으는 작업. 보톤 2,2부터 하는데 맘대로 하면 됨.
    
    # tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),# summery 보고싶으면 input_shapy 넣어야 한다. 여기선 28x28의 배열이다
    tf.keras.layers.Flatten(), # 데이터를 1차원 배열로 나열해준다. 왜쓰냐면 이거 없을 때 출력의 shape이 (None, 28,10)이어서
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
    # 정수 예측은 불가능하므로(안 쓴다) 확률 문제로 바꾸어서, 10개의 카테고리 각각의 확률을 뱉어준다
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# trainY(정답)이 정수면 sparse_categorical_crossentropy, 아니면 categorical_crossentropy 사용


model.summary() #요약본 나온다
# Output Shape
# (None, N, M) -> (입력 데이터 개수, ?, 노드 개수(설정한 값))
# Param은 train 가능한 가중치 개수


model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5) # 여기다 validation 추가해서 epoch 할 때마다 evaluate 할 수 있음
# model.fit( "(학습대상, 정답)인 길이가 2짜리인'튜플'", validation_data="왼쪽과 같은 '튜플'", epochs="반복횟수")

score = model.evaluate( testX, testY ) #모델 테스트임. 데이터는 학습 데이터 말고 새로운 걸 넣어야 함
print(score) # loss, accuracy가 출력됨


# validation_accuracy 높이기 위해 Dense를 더 만들든 conv2D + Pooling 더 하든 하면 됨
# train accuracy > test accuracy 이면 overfitting이라고 하고, 과도한 train으로 train 데이터만 더 잘 예측하는 것. 저렇게 부등호 되기 전에 epochs를 낮춰서 조정해야 함