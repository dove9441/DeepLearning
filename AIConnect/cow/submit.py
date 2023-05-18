import tensorflow as tf
import os
import pandas as pd
import numpy as np
from PIL import Image

load_model = tf.keras.models.load_model('./saved')

df = pd.DataFrame({ 'id' : [],
                  'grade' : []})


label_dic = {'1++' : 0,'1+' : 1, '1' : 2, '2' : 3, '3' : 4}

label_to_grade = {0 : '1++', 1 : '1+', 2 : '1', 3 : '2', 4 : '3'}


imgs = []
testY = []
filelist = os.listdir('./test/images/')

# 200*248
# 모델 학습 때와 동일한 방식으로 predict 해줘야 해서 expand_dims 해줌
flen = len(filelist)
for index, i in enumerate(filelist):
    # 사진 추가
    img = Image.open('./test/images/'+i).crop((30,52, 172,194)).resize((142,142))
    img = np.array(img)
    img = tf.expand_dims(img, axis=0)
    # predict 는 probability 배열을 제공하기 때문에, np.argmax를 사용해서 최대인 인덱스를 찾아야 한다
    pred_pos = load_model.predict(img)
    label = np.argmax(pred_pos[0])
    grade = label_to_grade[label]
    df.loc[index] = [i, grade]
    print(f'{index}/{flen} completed.')

    

    
df.to_csv('./result/result.csv', index=False)



