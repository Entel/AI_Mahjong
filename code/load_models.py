from game_simulation import GameSimulation as gs
from training_data_value import DataGenerator as dg
#from loss_point_nn import lossPointPredict as lpp
from waiting_tiles_nn import WaitingTilesPrediction as wtp 

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session

SHAPE_108 = [6, 6, 108] 
SHAPE_107 = [6, 6, 107] 


LOSS_POINT_MODEL = '../checkpoint/loss_point/weights_training.best.hdf5'
ZIMO_DATA = '../xml_data/zimo.dat'
WT_MODEL = '../model/waiting_tile.model'
WT_VAL = '../xml_data/wt_validation.dat'

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = 0.2,
        visible_device_list = '2'
    )
)
set_session(tf.Session(config=config))

def wt_data_generator_for_testing(datapath, shape):
    batch_x, batch_y = [], []
    count = 0
    with open(datapath) as f:
        for line in f:
            gen = gs.data_gen(line)
            item = gen[-1]
            x, y = dg.wt_data_gen(item)
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
        #self.lp = lpp()
        #self.lp_model = self.lp.create_model()

        self.wt = wtp()
        self.wt_model = self.wt.create_model()
        self.wt_model.load_weights(WT_MODEL)

    def waiting_tiles_pred(self, x):
        return self.wt_model.predict(np.reshape(x, [1, 6, 6, 107]))

    def waiting_tiles_evaluate(self, datapath):
        return self.wt_model.evaluate_generator(wt_data_generator_for_testing(datapath, SHAPE_107), steps=1000)

    '''
    def loss_point_pred(self, x):
        self.lp_model.load_weights('../checkpoint/loss_point/weights_without_zimo.best.hdf5')
        return self.lp_model.predict(np.reshape(x, [1, 6, 6, 108]))

    def loss_point_evaluate(self):
        self.lp_model.load_weights(LOSS_POINT_MODEL)
        return self.lp_model.evaluate_generator(zimo_data_generator(ZIMO_DATA), steps=1000)
    '''

if __name__ == '__main__':
    pred = Prediction()
    #print pred.loss_point_evaluate()
    pred.waiting_tiles_evaluate(WT_VAL)
    '''
    for test in testli:
        data = dg.data2tiles(test)
        data = dg.mesen_transfer(data)
        batch = dg.loss_data_gen(data)
        if batch[0]:
            loss_pred = pred.loss_point_pred(batch[1])
            print loss_pred
    '''
