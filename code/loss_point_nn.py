import numpy as np
import os
from game_simulation import GameSimulation as gs
from training_data_value import DataGenerator as dg

import keras
import tensorflow as tf
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
        visible_device_list = '1'
    )
)
set_session(tf.Session(config=config))

input_shape = (6, 6, 108)
SHAPE = [6, 6, 108]
epochs = 500
batch_size= 64
SUB_DATA_SIZE = 40000
nClasses = 9

TRAININGDATA = '../xml_data/lp_training.dat'
VALIDPATH = '../xml_data/lp_validation.dat'
LOSS_POINT_PATH = '../model/loss_point.model'
CHECKPOINT_PATH = '../checkpoint/loss_point/lp.improvement_{val_acc:.3f}.best.hdf5'
T_CHECKPOINT_PATH = '../checkpoint/loss_point/lp.t_improvement_{acc:.3f}.best.hdf5'

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
        self.t_checkpoint = ModelCheckpoint(T_CHECKPOINT_PATH, 
                                    monitor='acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')

    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(512, (4, 4), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(512, (2, 2), padding='same', name='layer_512', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (4, 4), padding='same'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='layer_256'))
        model.add(Dropout(0.5))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (4, 4), padding='same'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='layer_64'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(6, activation='softmax'))
    
        adam = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

        model.summary()
        return model

    def training(self):
        model = self.create_model()
        '''
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
                            
        '''
        for e in range(epochs):
            print('Epoch %d ---------------------------------------------' % e)
            os.system('shuf ' + TRAININGDATA + ' -o ' + TRAININGDATA)
            for X, Y in generate_data_from_file():
                model.fit(X, Y, 
                    epochs = 1,  
                    batch_size = batch_size, 
                    #validation_data = valid_data, 
                    validation_split = 0.3,
                    shuffle = True, 
                    callbacks = [self.tensorboard, self.checkpoint, self.t_checkpoint])

        model.save(LOSS_POINT_PATH)
        return model

    def pred(self, model_path):
        model = load_model(model_path)
        return model

def generate_data_from_file(path=TRAININGDATA, sub_data_size=SUB_DATA_SIZE):
    batch_x, batch_y = [], []
    count = 0
    #while True:
    with open(path) as f:
        for line in f:
            gen = gs.data_gen(line)
            item = gen[-1]

            x, y = dg.lp_data_gen(item)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append(y)
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
    batch_x, batch_y = [], []
    with open('../xml_data/fz_test.dat') as f:
        for line in f:
            gen = gs.data_gen_value(line) 
            for item in gen:
                pass
            data = data_process(item)
            x, y = dg.lp_data_gen(data)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append(y)
    '''
    lossPoint = lossPointPredict()
    model = lossPoint.training()
