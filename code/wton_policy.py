import numpy as np
from itertools import islice
from game_simulation import GameSimulation as gs
from training_data_policy import DataGenerator as dg
from training_data_policy import tile_matrix, HAI, MAX_TURN

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import Adam
from keras.utils import np_utils

input_shape = (6, 6, 107)
SHAPE = [6, 6, 107]
batch_size = 128
epochs = 20
nClasses = 4

DATAPATH = '../xml_record.dat'
WTON_PARAM_PATH = '../model/wether_waiting.model'
WT_PARAM_PATH = '../model/wether_waiting.model'



class waitingOrNot:
    def __init__(self):
        self.tensorboard = TensorBoard(log_dir = '../logs/waitingOrNot',
                                    histogram_freq = 0,
                                    write_graph = True,
                                    embeddings_freq = 0)
        self.checkpoint = ModelCheckpoint(CHECKPOINT_PATH, 
                                    monitor='val_acc', 
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
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(6))
        model.add(Activation('softmax'))
    
        adam = Adam(lr=10e-7)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.summary()
        return model

    def training(self):
        model = self.create_model()
        line_count = 0
        data_list = []
        step = 1
        batch_x = []
        batch_y = []
        with open(DATAPATH) as f:
            for line in f:
                if line_count != 64:
                    line_count += 1
                    data_list.append(line)
                else:
                    for data_line in data_list:
                        gen = gs.data_gen(data_line)
                        for item in gen:
                            waiting, x, wt, y = dg.generate_data_for_waiting(item)
                            if waiting:
                                batch_x.append(np.reshape(x, SHAPE))
                                batch_y.append(y)
                    model.fit(np.array(batch_x), np.array(batch_y), batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, callbacks=[self.tensorboard])
                    if step % 128 == 0:
                        model.save(LOSS_POINT_PATH)
                    step += 1

                    data_list = []
                    batch_x = []
                    batch_y = []
                    line_count = 0
                    data_list.append(line)

        model.save(WTON_PARAM_PATH)
        return model
             
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
