import tensorflow as tf
import matplotlib.pyplot as plt # 이미지를 파이썬에서 띄워보기 위해

#data = tf.keras.datasets.fashion_mnist.load_data() # 텐서플로우 기본 제공 데이터셋(쇼핑몰)
# 안의 내용물은 다음과 같다.
# ( ( trainX, trainY), (testX, testY) ) trainY는 정답값(아래의 class_names의 인덱스)이고, 왼쪽처럼 튜플로 들어있다.

#print(data[0][0].shape) # (60000,28,28)로 나온다. 28x28 이미지 6만개이다.

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(trainX[1])
# plt.gray() 흑백 출력
# plt.colorbar() 색상을 수치화해서 표시
# plt.show()
# pyplot은 주피터 노트북에서만 보인다..


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

### 모델 만들기
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),# summery 보고싶으면 input_shapy 넣어야 한다. 여기선 28x28의 배열이다
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(), # 데이터를 1차원 배열로 나열해준다. 왜쓰냐면 이거 없을 때 출력의 shape이 (None, 28,10)이어서
    tf.keras.layers.Dense(10, activation="softmax"),
    # 정수 예측은 불가능하므로(안 쓴다) 확률 문제로 바꾸어서, 10개의 카테고리 각각의 확률을 뱉어준다
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# trainY(정답)이 정수면 sparse_categorical_crossentropy, 정수면 categorical_crossentropy 사용


model.summary() #요약본 나온다
# Output Shape
# (None, N, M) -> (입력 데이터 개수, ?, 노드 개수(설정한 값))
# Param은 train 가능한 가중치 개수
model.fit(trainX, trainY, epochs=5)
