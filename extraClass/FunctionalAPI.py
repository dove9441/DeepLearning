import tensorflow as tf 
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)), 
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)


# 레이어 그림으로 미리 보는 코드
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)




### 순서대로인 Sequential Layer 말고 맘대로 순서 정하는 Functional Layer

input1 = tf.keras.layers.Input(shape=[28,28])
flatten1 = tf.keras.layers.Flatten()(input1) # 뒤에 괄호 하나 더 치고 이름쓰면 괄호 안 레이어 뒤에 위치하게 됨 (input1 -> flatten1)
dense1 = tf.keras.layers.Dense(28*28, activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape( (28,28 ))(dense1) # reshape 레이어는 모양 바꿔줌. reshape하려면 위의 노드 개수 = reshape한 노드 개수여야 함. 그래서 위의 노드도 28*28개
concat1 = tf.keras.layers.Concatenate()([input1, reshape1]) # 레이어 두개 합침
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten2)

functional_model = tf.keras.Model(input1, output)

functional_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

functional_model.fit(trainX, trainY, validation_data=(testX,testY), epochs=3)
plot_model(functional_model, to_file='functionalmodel.png', show_shapes=True, show_layer_names=True)
