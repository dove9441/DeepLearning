import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import time

label_dic = { '1++' : 0,'1+' : 1, '1' : 2, '2' : 3, '3' : 4 }


df = pd.read_csv('./train/grade_labels.csv')
# labelcon = df['imname'] == 'cow_1++_4754.jpg'
# label = df[labelcon].values[0][1]


# print(df['grade'].unique())
# exit()

imgs = []
trainY = []
filelist = os.listdir('./train/images/')
# 200*248
for i in filelist:
    # 사진 추가
    img = Image.open('./train/images/'+i).crop((30,52, 172,194)).resize((142,142))
    img = np.array(img)
    imgs.append(img)

    # Y값(등급) 추가
    labelcon = df['imname'] == i
    label = df[labelcon].values[0][1]
    trainY.append(label_dic[label])



imgs = np.array(imgs)
#imgs = imgs / 255.0
trainY = np.array(trainY)
print('completed.')


# 테스트 데이터 쪼개기
tX, vX, tY, vY = train_test_split(imgs, trainY, test_size=0.1, random_state=42)
# A, B, C, D일 때 A, C는 1-test_size만큼, B, D는 test_size만큼 갖는다. random_state는 그냥 42 쓴다 (이유는모름)
print('tX : %s, tY : %s, vX : %s, vY : %s' %(type(tX),type(tY),type(vX),type(vY)))



# 모델 만들기

model = tf.keras.Sequential([ # 200x248 크기인데 인식은 반대로 하나보다 inputshape오류남
    tf.keras.layers.Conv2D(32, (3,3), strides=(2, 2), padding='same', activation='relu', input_shape=(142,142,3)),
    # input_shape에는 '하나의 데이터' shape을 쓴다
    tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
	tf.keras.layers.Dropout(0.3),
    # pooling size는 대부분 2x2를 쓴다
    tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format('model'+str(int(time.time()))))
model.fit(tX, tY, batch_size=32,validation_data=(vX,vY), epochs=100, callbacks=[tensorboard])

# 텐서보드를 이용한 시각적 분석 tensorboard --logdir "로그 경로" --host=0.0.0.0 터미널에 입력하면 볼 수 있음



# EalryStopping을 이용하면 epochs가 많아도 알아서 overfitting이 일어나는 지점에서 멈춰줌 
model.save('./saved/')











