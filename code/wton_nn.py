import numpy as np
import random
import os
from itertools import islice
from game_simulation import GameSimulation as gs
from training_data_value import DataGenerator as dg

import keras
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.2,
        visible_device_list = '2'
    )
)
set_session(tf.Session(config=config))

input_shape = (6, 6, 107)
SHAPE = [6, 6, 107]
batch_size = 64
epochs = 150
SUB_DATA_SIZE = 40000

TRAININGDATA = '../xml_data/hozyu.dat'
VALIDATIONDATA = '../xml_data/wton_validation.dat'
WTON_PARAM_PATH = '../model/wton_nn.model'
CHECKPOINT_PATH = '../checkpoint/wton/wton.improvement_{val_loss:.3f}.hdf5'
T_CHECKPOINT_PATH = '../checkpoint/wton/wton.t_improvement_{loss:.3f}.hdf5'

class waitingOrNot:
    def __init__(self):
        self.tensorboard = TensorBoard(log_dir = '../logs/wton',
                                    histogram_freq = 0,
                                    write_graph = True,
                                    embeddings_freq = 0)
        self.checkpoint = ModelCheckpoint(CHECKPOINT_PATH, 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')
        self.t_checkpoint = ModelCheckpoint(T_CHECKPOINT_PATH, 
                                    monitor='loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')

    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
    
        adam = Adam(lr=10e-6)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

        model.summary()
        return model

    def training(self):
        model = self.create_model()

        '''
        with open(VALIDATIONDATA) as f:
            valid_x, valid_y = [], []
            lines = f.readlines()
            for line in lines:
                gen = gs.data_gen(line)
                x, y = dg.wton_data_gen(gen[-1])
                valid_x.append(np.reshape(x, SHAPE))
                valid_y.append(y)
        valid_data = (np.array(valid_x), np.array(valid_y))

        model.fit_generator(generator = generate_data_from_file(TRAININGDATA, batch_size),
            steps_per_epoch = 7500,
            epochs = epochs,
            validation_data = generate_data_from_file(VALIDATIONDATA, batch_size),
            validation_steps = 2437,
            use_multiprocessing = True,
            workers = 3,
            max_queue_size = 16,
            callbacks=[self.tensorboard, self.checkpoint, self.t_checkpoint])
        '''
        for e in range(epochs):
            print('Epoch %d =================================================' % e)
            os.system('shuf ' + TRAININGDATA + ' -o ' + TRAININGDATA)
            for X, Y in generate_data_from_file():
                model.fit(X, Y, 
                    epochs = 1,  
                    batch_size = batch_size, 
                    #validation_data = valid_data, 
                    validation_split = 0.3,
                    shuffle = True, 
                    callbacks = [self.tensorboard, self.checkpoint, self.t_checkpoint])

        model.save(WTON_PARAM_PATH)
        return model

def generate_data_from_file(path=TRAININGDATA, sub_data_size=SUB_DATA_SIZE):
    batch_x, batch_y = [], []
    count = 0
    #while True:
    with open(path) as f:
        for line in f:
            gen = gs.data_gen(line)
            item_1 = gen[-1]
            while True:
                item_2 = random.choice(gen[0:int(len(gen)*3/4)])
                if not item_2[-5]:
                    break

            x = dg.wton_data_gen(item_1)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append([1])
            count += 1

            x = dg.wton_data_gen(item_2)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append([0])
            count += 1

            if count == SUB_DATA_SIZE:
                yield np.array(batch_x), np.array(batch_y)
                count = 0
                batch_x = []
                batch_y = []
        if batch_x:
            yield np.array(batch_x), np.array(batch_y)

 
if __name__ == '__main__':
    '''
    with open('../xml_data/hozyu.dat') as f:
        for line in f:
            print(line)
            gen = gs.data_gen(line)
            item = gen[-1]
            break
        print(item)
    '''
    
    waiting_prediction = waitingOrNot()                
    model = waiting_prediction.training()
