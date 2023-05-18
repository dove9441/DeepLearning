import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('./train.csv')
# 데이터 쓰면 일단 null 있는지 확인
#print("before fill : \n",data.isnull().sum())

# 비어있으면 지우거나 다른 데이터로 채워넣거나. 지울 게 많으니 평균값 넣기
ageavr = data['Age'].mean()
# print(ageavr)

# null값을 30으로 채움 
data['Age'].fillna(value=30, inplace=True) #inplace 안쓰면 원본 유지돼서 변수로 새로운데이터 받아와야함

# Embarked도 채워야 하는데 이번엔 최빈값을 넣을거임 
emb = data['Embarked'].mode() # 최빈값 뽑아줌
# print(emb) 
data['Embarked'].fillna(value='S', inplace=True)

#print("after fill : \n",data.isnull().sum())

# 데이터셋 자료를 만들어줌
# tf.data.Dataset.from_tensor_slices( (X데이터, 정답) ) 튜플로넣어
trainY = data.pop('Survived') #이러면 해당 열만 나옴. (원본 반환 X)
ds = tf.data.Dataset.from_tensor_slices( (dict(data), trainY) ) #이런 데이터셋 만들어야 featurecol쓸수있음

# for i, l in ds.take(1):
#     print(i,l)



# feature column 이용하기

feature_columns = []

# 이러면 fare 0~1로 압축됨 무슨 수학적 원리인지는 모름
def NF(x):
    dmin = data['Fare'].min()
    dmax = data['Fare'].max()
    return (x-dmin)/(dmax-dmin) 


### 숫자로 넣을 거 SibSp Parch	Fare : numeric_column
feature_columns.append( tf.feature_column.numeric_column('Fare', normalizer_fn=NF) )
feature_columns.append( tf.feature_column.numeric_column('Parch') )
feature_columns.append( tf.feature_column.numeric_column('SibSp') )

### 카테고리로 넣을 거(원 핫 인코딩) Sex Embarked Pclass : indicated_column
cat_list = ['Sex','Embarked','Pclass']
for i in cat_list:
    vocab = data[i].unique() 
    cat = tf.feature_column.categorical_column_with_vocabulary_list(i, vocab) # ( 열 이름, unique한 문자 리스트) (직접 [male, female] 이라고 넣어도 됨)
    one_hot = tf.feature_column.indicator_column(cat) # 이러면 원 핫 인코딩 해줌
    feature_columns.append(one_hot)
    #위의 4개가 전처리하는 법임

#print(feature_columns)

### 카테고리(개많음, 원핫 x) Ticket : embedding_column (전에했음)
vocab = data['Ticket'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket',vocab)
embeded = tf.feature_column.embedding_column(cat, dimension=9) # dimension은 맘대로 하면 됨
feature_columns.append(embeded)




### 범주화해서 넣을 거 Age(10대, 20대 등) : bucketized_column
Age = tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10,20,30,40,50,60]) # 이러면 범주화 알아서 해줌
feature_columns.append(Age_bucket)








# 모델 만들기

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),  # feature_column 쓰려면 첫 번째 레이어는 무조건 이거임
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# batch 안뽑아서 넣으면 오류난대요
ds_batch = ds.batch(32)


model.fit(ds_batch, shuffle=True, epochs=20)
# Y데이터는 데이터셋활용할 떄 넣으면에러남 ValueError: `y` argument is not supported when using dataset as input.

