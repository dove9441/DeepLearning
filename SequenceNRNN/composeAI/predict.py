import tensorflow as tf

Pmodel = tf.keras.models.load_model('./200model')

import numpy as np


text = open('./pianoabc.txt','r').read()

textclass = list(set(text)) # 집합에 넣었다가 다시 리스트로 변환하면 중복이 제거되기 때문에 unique한 단어장을 만들 수 있다
textclass.sort() # 깔끔하라고


# utilities
text_to_num = {}
num_to_text = {}

for i, data in enumerate(textclass): #enumerate 쓰면 (인덱스, 데이터) 이런 걸 뱉어서 파라미터 두개로 각각 뽑을 수 있음
    text_to_num[data] = i
    num_to_text[i] = data # 이렇게 둘 다 만들어놓으면 매우 유용함
    
num_text_list = [] 
    
for i in text:
    num_text_list.append(text_to_num[i])
    
#####################################
 
first = num_text_list[0:0+25]
first = tf.one_hot(first, 31)
first = tf.expand_dims(first, axis=0) # shape 안 맞아서 차원늘림


music = []
for i in range(200):
    pred = Pmodel.predict(first)
    # print(pred) # 하면 31개의 원 핫 인코딩된 문자 각각일 확률을 보여줌.
    pred = np.argmax(pred[0]) # print해봤을 때 list 안에 list가 하나 있어서 list 하나만 넣기 위해 인덱싱
    new_pred = np.random.choice(textclass,1,pred[0]) # 이러면 textclass 중 1개를 pred[0]만큼의 확률로 뽑아줌
    # print("예측 : ",num_to_text[pred])



    music.append(pred)
    next=first.numpy()[0][1:] # 리스트 슬라이싱 이용하기 위해 tensor를 numpy 배열로 바꿔줌. 저렇게 하면 처음 하나 없애줌. 지금 3중 리스트이기 때문에 앞에 [0]해주는 것
    one_hot_num = tf.one_hot(pred,31)
    first = np.vstack([ next, one_hot_num.numpy()]) # vstack이 리스트 두 개 합쳐줌
    first = tf.expand_dims(first, axis=0)

    
music_text = []
for i in music:
    music_text.append(num_to_text[i])
    
print(''.join(music_text))