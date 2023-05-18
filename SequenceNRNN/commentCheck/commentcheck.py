# 데이터 다운로드 (네이버 쇼핑 평점과 댓글로 구성되어있음)
# import urllib.request
# urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', 'shopping.txt')
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

raw = pd.read_table('./shopping.txt', names=['rating', 'review']) 
# 데이터프레임 만들기 위해 열의 제목이 필요한데 없어서 이렇게 꼭 제목을 달아줘야 한다. read_table은 자동으로 구분자를 인식해서 데이터프레임을 만들어준다


# Y데이터(정답)은 rating이라 생각할 수 있지만 0,1 이런 식으로 악평 / 호평을 구분할 수 있게 라벨링을 해 줘야 한다.

raw['label'] = np.where( raw['rating']> 3, 1, 0) # where은 조건부 함수인데, 삼항 연산자처럼 'rating' row가 3보다 크면 1, 아니면 0으로 새로운 열을 추가해준다


# 한글 데이터 전처리하는 법 : 정제되지 않은 한글 데이터는 각 '문자'를 1, 2,3 ...등으로 저장하는 게 좋다. 
# 특수문자는 쓸데없으므로 다 제거해준다

## 데이터 전처리
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|0-9 ]', '') # 정규 표현식에 맞지 않는 것은 제거. 공백 추가하면 공백도 남김

# raw.isnull() 만약 위에서 전처리했는데 다 바뀌어서 아무 것도 없는 공백이면 아예 없애야 함
raw.drop_duplicates( subset=['review'], inplace=True) # 중복 제어. subset에는 검사할 열 '이름', inplace=True 안 쓰면 원본 데이터프레임을 유지하고 중복 제거된 데이터를 리턴한다.


### unique한 문자 모으기
unique_text = raw['review'].tolist() # 리스트에 넘김
unique_text = ''.join(unique_text) # 리스트 안에 담긴 문자 모두 붙임
unique_text = list(set(unique_text)) # 중복 제거됨
unique_text.sort()
#print(unique_text)



#### 한글 숫자로 변환
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
# 난 텍스트를 정수로 바꾸겠다 하는 함수임. char_level false면 단어 단위로 바꿈. oov_token은 out of vocabulary이며 단어사전에 없는 단어를 해당 단어로 바꾸어줌. 저렇게 쓰는 게 일반적

raw_review_list = raw['review'].tolist()
tokenizer.fit_on_texts(raw_review_list) # 집어넣은 데이터에 있는 모든 문자들을 정수로 바꿔줌
#print(tokenizer.word_index) # word dicionary 알아서 만들어줌

# 팁 : 단어단위로 전처리할 경우 전체 데이터 중 1회 이하 출현하는 단어는 dicionary에서 삭제하면 성능이 향상될 수 있음

# train dataset 숫자로 변환하기
train_seq = tokenizer.texts_to_sequences(raw_review_list)

##### 모델에 집어넣기 위해서는 글자 길이가 일치해야 함

# 문장의 길이 열 만들기
raw['length'] = raw['review'].str.len()
# print(raw.head())
# print(raw.describe()) #length 열 만들고 describe 해보면 최대 length 알 수 있음

#print(len(raw.loc[raw['length']<100])) # 100자 미망 세보니 19만개 / 19.9만개 이여서 100자 제한하기로함

# 100자 제한하기. 제한 뿐만 아니라 pad로 길이를 맞추어준다. 안하면 오류난다.
# 아래 두 개 쓰면 그냥 알아서 해줌
from tensorflow.keras.preprocessing.sequence import pad_sequences
trainX = pad_sequences(train_seq, maxlen=100)


from sklearn.model_selection import train_test_split

# 정답 데이터 만들기
trainY = raw['label'].tolist()

# 테스트 데이터 쪼개기
tX, vX, tY, vY = train_test_split(trainX, trainY, test_size=0.2, random_state=42) 
# A, B, C, D일 때 A, C는 1-test_size만큼, B, D는 test_size만큼 갖는다. random_state는 그냥 42 쓴다 (이유는모름)



### 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(unique_text)+1, 16), 
    # Embedding(vocab_size, embedding_dim, input_length=5)
    # 원 핫 인코딩은 unique한 문자 개수만큼의 비트를 갖는데, 이건 unique한 문자 개수만 넣으면 16자리의 무작위 숫자가 담긴 행렬로 바꿔줌. +1은 OOV용. embed는 입력에만 써야함
    tf.keras.layers.LSTM(100, return_sequences=True), #LSTM 쌓을 거면 return_sequences써줘야함
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# 안하면 오류나더라 ? ? 
tX = np.array(tX)
tY = np.array(tY)
vX = np.array(vX)
vY = np.array(vY)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(tX, tY, validation_data=(vX,vY), batch_size=1000, epochs=20)
model.save('./model1')

