import os
import csv
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_process

from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, TensorBoard
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import RMSprop
from keras.utils import np_utils

from gensim.models import word2vec

from sklearn.decomposition import PCA

from scipy.stats import pearsonr

emb_param = '../param2.hdf5'
filename = '../final_hai.csv'
input_dim = 34
vec_dim = 34
output_dim = 34
batch_size = 1024
epochs = 10

HAI = ['1m',
    '2m',
    '3m',
    '4m',
    '5m',
    '6m',
    '7m',
    '8m',
    '9m',
    '1p',
    '2p',
    '3p',
    '4p',
    '5p',
    '6p',
    '7p',
    '8p',
    '9p',
    '1s',
    '2s',
    '3s',
    '4s',
    '5s',
    '6s',
    '7s',
    '8s',
    '9s',
    'Tou',
    'nan',
    'sya',
    'pei',
    'hak',
    'hat',
    'cyu',
    ]

class prediction:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensorboard = TensorBoard(log_dir='../logs', 
                                    histogram_freq = 0,
                                    write_graph = True,
                                    embeddings_freq = 0)
        self.data_generation = data_generation()

    def creat_model(self):
        model = Sequential()
        model.add(Embedding(self.input_dim, self.output_dim, input_length=vec_dim))
        model.add(Flatten())
        model.add(Dense(self.input_dim, use_bias=False, kernel_initializer=glorot_uniform(seed=20170719)))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['categorical_accuracy'])
        print(model.summary())
        return model

    def train(self, batch_size, epochs, emb_param):
        model = self.creat_model()
        data_list = data_generation.read_csv(filename)
        i = 1
        turn = []
        for data in data_list:
            if i % 10000 == 0:
                x, y = self.data_generation.gen_batch(turn)
                x = np.reshape(x, [len(x), input_dim])
                y = np.reshape(y, [len(y), input_dim])
                turn = []
                i += 1
                model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, callbacks=[self.tensorboard])
            else:
                turn.append(data)
                i += 1
        
        model.save_weights(emb_param)
        return model

    def load_weight(self):
        model = self.creat_model()
        try:
            model.load_weights(emb_param)
            return model.get_weights()
        except:
            return False

class data_generation:
    def __init__(self):
        self.filename = ''    

    def gen_x(self, data):
        for i in data:
            self.x = [0] * input_dim
            self.x[int(i)] = 1
            yield np.array(self.x)

    def gen_y(self, data):
        self.y = [0] * output_dim
        for i in data:
            self.y[int(i)] = 1
        return np.array(self.y)

    def gen_batch(self, turn):
        batch_x = []
        batch_y = []
        for data in turn:
            x_list = self.gen_x(data)
            item_y = self.gen_y(data)
            for batch in x_list:
                batch_x.append(batch)
                batch_y.append(item_y)
        return batch_x, batch_y
    
    def read_csv(self, filename):
        with open(filename) as csvfile:
            data_list = csv.reader(csvfile)
            for data in data_list:
                yield sorted(data)

class PCA_process:
    def __init__(self, data):
        self.data_process = data_process.data_process('', '')
        self.data = data
        self.pca = PCA(n_components=2)
    
    def pca_transform(self):
        px, label = [], []
        x, y = [], []
        for i in range(self.data.shape[0]):
            px.append(self.data[i])
            label.append(self.data_process.num2tiles(i))

        reduce_x = self.pca.fit_transform(px)

        for i in range(len(reduce_x)):
            x.append(reduce_x[i][0])
            y.append(reduce_x[i][1])

        return x, y, label

    def plt_color(self, num):
        c = float(num + 1) / 36.0
        return c

    def avg_cal(self, x, y, label):
        _x, _y = [], []
        for i in set(label):
            j = 4 * i
            _x.append((x[j]+x[j+1]+x[j+2]+x[j+3])/4)
            _y.append((y[j]+y[j+1]+y[j+2]+y[j+3])/4)
        return _x, _y, list(set(label))

    def show_graph(self):
        x, y, label = self.pca_transform()
        #x, y, label = self.avg_cal(x, y, label)
        for i in range(len(x)):
            plt.scatter(x[i], y[i], marker='$'+HAI[i]+'$', s=250)
            '''
            if label[i]<27 and label[i]%9 < 5:
                plt.scatter(x[i], y[i], c='r')
            elif label[i]<27 and label[i]%9 >= 5:
                plt.scatter(x[i], y[i], c='g')
            else:
                plt.scatter(x[i], y[i], c='b')
            ''' 
        plt.show()

class data_iterator:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        with open(self.dirname) as csvfile:
            data_list = csv.reader(csvfile)
            for data in data_list:
                yield sorted(data)
        

if __name__ == '__main__':
    '''
    prediction = prediction(input_dim, output_dim)
    data_generation = data_generation()
    
    #model = prediction.train(batch_size, epochs, emb_param)

    param = prediction.load_weight()[0]
    pca = PCA_process(param)
    pca.show_graph()
    '''
    '''
    '''
    sentences = data_iterator(filename)
    model = word2vec.Word2Vec(sentences, size=200, window=3)
    
    model.save('w2v_w3.model')

    model = word2vec.Word2Vec.load('w2v_w3.model')
    w2v_re = []
    for i in range(0, 34):
        w2v_re.append(model[str(i)])

    pca = PCA_process(np.array(w2v_re))
    pca.show_graph()
