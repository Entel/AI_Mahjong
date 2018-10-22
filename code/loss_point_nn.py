import numpy as np
import os
from game_simulation import GameSimulation as gs
from training_data_value import DataGenerator as dg

import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.2
        visible_device_list = '1',
    )
)
set_session(tf.Session(config=config))

input_shape = (6, 6, 108)
SHAPE = [6, 6, 108]
epochs = 1000
batch_size = 32

DATAPATH = '../xml_data/lp_rd_collect_data.dat'
TRAININGPATH = '../xml_data/lp_training.dat'
VALIDPATH = '../xml_data/lp_validation.dat'
LOSS_POINT_PATH = '../model/loss_point_without_zimo.model'
CHECKPOINT_PATH = '../checkpoint/loss_point/weights.best.hdf5'
T_CHECKPOINT_PATH = '../checkpoint/loss_point/weights_training.best.hdf5'

class lossPointPredict:
    def __init__(self):
        self.tensorboard = TensorBoard(log_dir = '../logs/loss_point',
                                    histogram_freq = 0,
                                    write_graph = True,
                                    write_images = True,
                                    embeddings_freq = 0)
        self.checkpoint = ModelCheckpoint(CHECKPOINT_PATH, 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')
        self.checkpoint_training = ModelCheckpoint(T_CHECKPOINT_PATH, 
                                    monitor='acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')

    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(512, (2, 2), padding='same', input_shape=input_shape))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (4, 4), padding='same', activation='relu', name='layer_512'))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='layer_256'))
        model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='layer_64'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(9, activation='softmax'))
    
        adam = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

        model.summary()
        return model

    def training(self):
        model = self.create_model()
        line_count = 0
        data_list = []
        batch_x = []
        batch_y = []
        valid_x = []
        valid_y = []

        with open(VALIDPATH) as f:
            lines = f.readlines()
            for line in lines:
                gen = gs.data_gen_value(line)
                for item in gen:
                    pass
                x, y = dg.lp_data_gen(item)
                valid_x.append(np.reshape(x, SHAPE))
                valid_y.append(y)

        with open(TRAININGPATH) as f:
            for line in f:
                gen = gs.data_gen_value(line) 
                for item in gen:
                    pass
                x, y = dg.lp_data_gen(item)
                batch_x.append(np.reshape(x, SHAPE))
                batch_y.append(y)
        model.fit(np.array(batch_x), np.array(batch_y), batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(np.array(valid_x), np.array(valid_y)) ,shuffle=True, callbacks=[self.tensorboard, self.checkpoint, self.checkpoint_training])
        #model.fit(np.array(batch_x), np.array(batch_y), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, shuffle=True, callbacks=[self.tensorboard, self.checkpoint, self.checkpoint_training])
                            
        model.save(LOSS_POINT_PATH)
        return model

    def pred(self, model_path):
        model = load_model(model_path)
        return model

if __name__ == '__main__':
    '''
    batch_x, batch_y = [], []
    with open('../xml_data/fz_test.dat') as f:
        for line in f:
            print line
            gen = gs.data_gen_value(line) 
            for item in gen:
                pass
            print item
            data = data_process(item)
            print data
            print item
            x, y = dg.lp_data_gen(data)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append(y)
    '''
    lossPoint = lossPointPredict()
    model = lossPoint.training()
