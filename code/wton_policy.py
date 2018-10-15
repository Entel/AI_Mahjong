import numpy as np
from itertools import islice
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
        per_process_gpu_memory_fraction=0.2,
        visible_device_list = '1'
    )
)
set_session(tf.Session(config=config))

input_shape = (6, 6, 107)
SHAPE = [6, 6, 107]
batch_size = 32
epochs = 1000
nClasses = 4

TRAININGDATA = '../xml_data/wton_training.dat'
VALIDATIONDATA = '../xml_data/wton_validation.dat'
WTON_PARAM_PATH = '../model/wether_waiting.model'
CHECKPOINT_PATH = '../checkpoint/waiting_or_not/wton.improvement_{epoch:02d}_{val_acc:.3f}.hdf5'
T_CHECKPOINT_PATH = '../checkpoint/waiting_or_not/wton.t_improvement_{epoch:02d}_{acc:.3f}.hdf5'

class waitingOrNot:
    def __init__(self):
        self.tensorboard = TensorBoard(log_dir = '../logs/waiting_or_not',
                                    histogram_freq = 0,
                                    write_graph = True,
                                    embeddings_freq = 0)
        self.checkpoint = ModelCheckpoint(CHECKPOINT_PATH, 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')
        self.t_checkpoint = ModelCheckpoint(CHECKPOINT_PATH, 
                                    monitor='acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='auto')

    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(512, (2, 2), padding='same', activation='relu', input_shape=input_shape))
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
        model.add(Dense(3))
        model.add(Activation('softmax'))
    
        adam = Adam(lr=10e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.summary()
        return model

    def training(self):
        model = self.create_model()
        class_weight = {0: 77113, 1: 80232, 2: 85610}

        model.fit_generator(generator = generate_data_from_file(TRAININGDATA, batch_size),
            steps_per_epoch = 7500,
            epochs = epochs,
            validation_data = generate_data_from_file(VALIDATIONDATA, batch_size),
            validation_steps = 2437,
            class_weight = class_weight,
            use_multiprocessing = True,
            workers = 3,
            max_queue_size = 16,
            callbacks=[self.tensorboard, self.checkpoint, self.t_checkpoint])

        model.save(WTON_PARAM_PATH)
        return model

def generate_data_from_file(path, batch_size):
    batch_x, batch_y = [], []
    count = 0
    while True:
        with open(path) as f:
            for line in f:
                gen = gs.data_gen_value(line)
                for item in gen:
                    pass
                x, y = dg.wton_data_gen(item)
                batch_x.append(np.reshape(x, SHAPE))
                batch_y.append(y)
                count += 1
                if count == batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    count = 0
                    batch_x = []
                    batch_y = []


 
if __name__ == '__main__':
    '''
    with open('../test.dat') as f:
        for line in f:
            gen = gs.data_gen(line)
            for item in gen:
                waiting, x, wt, y = generate_data_for_waiting(item)
                if waiting:
                    print np.array(x).shape, wt, y
            break
    print np.reshape(x, SHAPE)[0]
    '''
    
    waiting_prediction = waitingOrNot()                
    model = waiting_prediction.training()
