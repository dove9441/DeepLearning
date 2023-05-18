import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image # 이미지 전처리
import os
import matplotlib.pyplot as plt # 이미지 미리보기 하기 위해


filelist = os.listdir('./img_align_celeba/img_align_celeba')

imgs = []

##### 이미지 전처리

for i in filelist[0:50000]:
    img = Image.open('./img_align_celeba/img_align_celeba/'+i).crop( (20,30,160,200) ).convert('L').resize( (64,64) ) # resize랑 convert 안하면 메모리초과남,,
    imgs.append( np.array(img) ) # 이미지의 여백(얼굴 자체를 제외한 부분)을 지우기 위해 크롭. 좌측 상단을 0,0으로 잡고 아래로 갈수록 증가하는 '-y'축을 사용한다
    # 0~1사이로 압축(나누기) 하기 위해 np array로 바꾸기. np쓰면 행렬 전체 나누기 가능 .resize( (64,64) ) 하면 resize됨, .convert(L) 하면 흑백됨(시간절약)

# plt.imshow(imgs[40])
# plt.show 주피터 노트북에서 봐야함

imgs = np.divide(imgs, 255) # 0~1 사이로 압축(시간절약)(나누기)
imgs = imgs.reshape( 50000, 64, 64, 1 ) # tf모델에는 4차원 행렬이 필요해서 마지막 1추가하기 위해 reshape
print('shape : ',imgs.shape)



##### 모델 만들기. discriminator는 이미지를 사람인지 아닌지 판별해주는 모델
# genetator는 숫자 100개 넣으면 사람같은이미지 뱉어주는 모델
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=[64,64,1]),
    # stride는 필터를 적용하는 간격을 말한다. 이미지처리는 그냥 relu쓴다고만 알자
    # 근데 LeakyReLU는 음수값을 0.01 곱해서 0이 아닌 작은 값으로만 해줌 이거쓰는게 결과 잘 나온대
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

noise_shape = 100

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ), 
  tf.keras.layers.Reshape((4, 4, 256)),
  # Conv2DTranspose 쓰면 이미지가 점점 커짐. conv류는 이미지를 입력해줘야됨
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])


GAN = tf.keras.models.Sequential([ generator, discriminator ])

discriminator.compile( optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False # generator 트레이닝 시킬 때 discriminator는 학습하면 안됨
GAN.compile( optimizer='adam', loss='binary_crossentropy')




def predict_pic():
    random = np.random.uniform(-1, 1, size=(10,100)) # 100개 숫자를 10번 -1~1 사이로 뽑아줌
    predict = generator.predict(random)
    for k in range(10):
        plt.subplot(2,5, k+1)
        plt.imshow(predict[k].reshape(64,64), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


trainX = imgs
for j in range(300):
    print(f'epoch {j}/300 processing ...')
    predict_pic()
    for i in range(50000//128):
        if i%100==0:
            print(f'batch {i} processing...')
        ### discriminator 트레이닝

        # 진짜사진 트레이닝
        realimgs = trainX[i*128 : (i+1)*128]
        ones = np.ones(shape=(128,1)) # 진짜 사진은 1로 마킹되어있어야 하므로 1이 가득찬 리스트 생성
        loss1 = discriminator.train_on_batch(realimgs, ones) # train_on_batch는 loss값을 반환함
        # 가짜사진 트레이닝
        random = np.random.uniform(-1, 1, size=(128,100))
        fakeimgs = generator.predict(random)
        zeros = np.zeros(shape=(128,1))
        loss2 = discriminator.train_on_batch(fakeimgs,zeros)


        ### generator 트레이닝
        np.random.uniform(-1, 1, size=(128,100))
        ones = np.ones(shape=(128,1))
        loss3 = GAN.train_on_batch(random, ones)
    print(f'epoch {j} loss -> Discriminator : {loss1+loss2}, GAN : {loss3}')

discriminator.save('./models/discriminator')
GAN.save('./models/GAN')

