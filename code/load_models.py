from game_simulation import GameSimulation as gs
from training_data_value import DataGenerator as dg
from loss_point import lossPointPredict as lpp
from waiting_tiles_nn import WaitingTilesPrediction as wtp 

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import Adam
from keras.utils import np_utils

TEN_MATRIX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 18, 24]
108_SHAPE = [6, 6, 108] 
107_SHAPE = [6, 6, 107] 


LOSS_POINT_MODEL = '../checkpoint/loss_point/weights_training.best.hdf5'
ZIMO_DATA = '../xml_data/zimo.dat'


def data_generator_for_testing(datapath, shape):
    batch_x, batch_y = [], []
    count = 0
    with open(datapath) as f:
        for line in f:
            gen = gs.data_gen_value(line)
            for item in gen:
                pass
            x, y = dg.lp_data_gen(item)
            batch_x.append(np.reshape(x, shape))
            batch_y.append(y)
            count += 1
            
            if count == 64:
                yield np.array(batch_x), np.array(batch_y)
                count = 0
                batch_x = []
                batch_y = []

class Prediction:
    def __init__(self):
        #self.lp_model = load_model(LOSS_POINT_MODEL)
        self.lp = lpp()
        self.lp_model = self.lp.create_model()

        self.wt = wtp()
        self.wt_model = self.wt.create_model()
        self.wt_model.load_weights(WT_MODEL)

    def waiting_tiles_pred(self, x):
        return self.wt_model.predict(np.reshape(x, [1, 6, 6, 107]))

    def waiting_tiles_evaluate(self, datapath):
        return self.wt_model.evaluate_generator(data_generator_for_testing(datapath, 107_SHAPE), steps=1000)

    def loss_point_pred(self, x):
        self.lp_model.load_weights('../checkpoint/loss_point/weights_without_zimo.best.hdf5')
        return self.lp_model.predict(np.reshape(x, [1, 6, 6, 108]))

    def loss_point_evaluate(self):
        self.lp_model.load_weights(LOSS_POINT_MODEL)
        return self.lp_model.evaluate_generator(zimo_data_generator(ZIMO_DATA), steps=1000)

if __name__ == '__main__':
    pred = Prediction()
    print pred.loss_point_evaluate()
    '''
    for test in testli:
        data = dg.data2tiles(test)
        data = dg.mesen_transfer(data)
        batch = dg.loss_data_gen(data)
        if batch[0]:
            loss_pred = pred.loss_point_pred(batch[1])
            print loss_pred
    '''
