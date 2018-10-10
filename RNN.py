import pickle
from os.path import join
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

def SaveModel(clf):
    filename = 'QUALITY_new_RNN.pkl'
    saved_model = open(join("models",filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()

def LoadData(path_data,path_label):
    datas = []
    labels = []
    with open(path_data, 'r', encoding='utf-8')as file:
        for i in file:
            datas.append(i)

    with open(path_label, 'r', encoding='utf-8')as file:
        for i in file:
            labels.append(i)
    return datas, labels


def SaveModel(clf):
    filename = 'LOCATION_new.pkl'
    saved_model = open(join("models",filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()

datas = []
categories = []
datas_valid = []
categories_valid = []

with open(join("data_train", "datas_QUALITY_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_QUALITY_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        categories.append(i)

with open(join("data_valid", "datas_QUALITY_valid_new.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)

with open(join("data_valid", "labels_QUALITY_valid_new.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        categories_valid.append(i)


df = pd.DataFrame({"datas": datas, "categories": categories})
df1 = pd.DataFrame({"datas_valid": datas_valid, "categories_valid": categories_valid})
data = df['datas']
label = df['categories']
data_valid = df1['datas_valid']
label_valid = df1['categories_valid']

X_train = data
Y_train = label
X_valid = data_valid
Y_valid = label_valid

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_train = Y_train.reshape(-1,1)
Y_valid = le.fit_transform(Y_valid)
Y_valid = Y_valid.reshape(-1,1)



max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
print(model.summary())
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2)



