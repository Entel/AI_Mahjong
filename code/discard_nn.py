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
        per_process_gpu_memory_fraction = 0.2,
        visible_device_list = '2'
    )
)
set_session(tf.Session(config=config))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

input_shape = (6, 6, 107)
SHAPE = [6, 6, 107]
batch_size = 64
SUB_DATA_SIZE = 20000
epochs = 3000

TRAININGDATA = '../data/discard_training.dat'
VALIDATIONDATA = '../data/discard_validation.dat'
DT_PARAM_PATH = '../model/discard_tile.model'
CHECKPOINT_PATH = '../checkpoint/discard_tile/discard_tiles.improvement_{epoch:02d}_{val_acc:.3f}.hdf5'
T_CHECKPOINT_PATH = '../checkpoint/discard_tiles/discard_tiles.t_improvement_{epoch:02d}_{acc:.3f}.hdf5'

class DiscardTile:
    def __init__(self):
        self.tensorboard = TensorBoard(log_dir = '../logs/discard_tile',
                                    histogram_freq = 0,
                                    write_graph = True,
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
        
        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Conv2D(512, (4, 4), padding='same', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (2, 2), padding='same'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(34))
        model.add(Activation('softmax'))
    
        adam = Adam(lr=10e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.summary()
        return model    

    def training(self):
        model = self.create_model()

        '''
        with open(VALIDATIONDATA) as f:
            valid_x, valid_y = [], []
            lines = f.readlines()
            for line in lines:
                gen = gs.data_gen_value(line)
                for item in gen:
                    pass
                x, y = dg.wt_data_gen(item)
                valid_x.append(np.reshape(x, SHAPE))
                valid_y.append(y)
        valid_data = (np.array(valid_x), np.array(valid_y))

        model.fit_generator(generator = generate_data_from_file(TRAININGDATA, batch_size),
            steps_per_epoch = 3400,
            epochs = epochs,
            validation_data = (np.array(valid_x), np.array(valid_y)),
            #use_multiprocessing = True,
            #workers = 3,
            #max_queue_size = 10,
            callbacks=[self.tensorboard, self.checkpoint, self.t_checkpoint])
        '''
        for e in range(epochs):
            print('Epoch %d' % e)
            os.system('shuf ' + TRAININGDATA + ' -o ' + TRAININGDATA)
            for X, Y in generate_data_from_file():
                model.fit(X, Y, 
                    epochs = 1,  
                    batch_size = batch_size, 
                    #validation_data = valid_data, 
                    validation_split = 0.2,
                    shuffle = True, 
                    callbacks = [self.tensorboard, self.checkpoint, self.t_checkpoint])

        model.save(WT_PARAM_PATH)
        return model
             
def generate_data_from_file(path=TRAININGDATA, sub_data_size=SUB_DATA_SIZE):
    batch_x, batch_y = [], []
    count = 0
    #whipple True:
    with open(path) as f:
        for line in f:
            x, y = dg.discard_data_gen(line)
            batch_x.append(np.reshape(x, SHAPE))
            batch_y.append(y)
            count += 1
            if count == sub_data_size:
                yield np.array(batch_x), np.array(batch_y)
                count = 0
                batch_x = []
                batch_y = []
            else:
                if not batch_x:
                    yield np.array(batch_x), np.array(batch_y)


if __name__ == '__main__':
    '''
    for x, y in generate_data_from_file('../xml_data/fz_test.dat', 3):
        print y
            
    '''
    dt = DiscardTile()
    model = dt.training()
