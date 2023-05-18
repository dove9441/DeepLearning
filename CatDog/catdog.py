import tensorflow as tf
import os
import shutil  # 파일 옮기기 쉽게 하기 위한 모듈



for i in os.listdir('./dogs-vs-cats-redux-kernels-edition'):
    if 'cat' in i:
        shutil.copyfile('./dogs-vs-cats-redux-kernels-edition/train/train'+i, './dataset/cat/'+i) # 대상 위치의 파일 -> 이동할 위치의 파일
    else:
        shutil.copyfile('./dogs-vs-cats-redux-kernels-edition/train/train'+i, './dataset/dog/'+i)