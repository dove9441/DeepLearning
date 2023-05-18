import tensorflow as tf
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
    
# 25개씩 x, y 만들기 trainX는 앞 25개이고, trainY는 정답값(예측값)이므로 앞에서 이어지는 25개 다음 1개를 쓰면 된다.

trainX = []
trainY = []
temp = []

for i in range(0,len(num_text_list)-25):
    trainX.append(num_text_list[i:i+25]) # 리스트 슬라이싱 [n,m]은 list[n] ~ list[m-1] 까지 가져옴
    trainY.append(num_text_list[i+25])
        
        
# 원 핫 인코딩하기(안하고 나온 리스트 그대로 넣어도 됨 잘나오는거 선택)
print(len(textclass)) # -> 종류의 개수가 31개임. 따라서 길이 31짜리 배열로 원 핫 인코딩하면됨

# 배열만들어서 코드짜도되는데 tf에 내장함수있음ㅋㅋ
X = tf.one_hot(trainX,31) # tf.one_hot(array, class개수)
Y = tf.one_hot(trainY,31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25,31), return_sequences=True), # layers.TYPE(node개수, input_shpe(하나의 데이터의 shape, 첫 번째 레이어만))
    # LSTM 여러 개 쌓을 거면 저거 바꿔줘야함 왠지는 아직 이해안됨,  LSTM, GRU는 activation 없음(이미 정해져있음
    tf.keras.layers.LSTM(100), 
    tf.keras.layers.Dense(31, activation="softmax"),
])


model.compile(loss='categorical_crossentropy', optimizer='adam') # one hot 안되어있으면 앞에 sparse 붙어야함
model.fit(X,Y, batch_size=100, epochs=200, verbose=1)
          # batch_size만큼의 데이터를 넣은 후에 w값을 업데이트한다. LSTM은 epochs많이필요함. (10이상)
        #0 = silent, 1 = default(progress bar + loss + accuracy), 2 = loss,accuracy만, 3 = only epochs
        
model.save('./200model')
