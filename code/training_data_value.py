import numpy as np
import copy
import math
from game_simulation import GameSimulation as gs
from training_data_policy import DataGenerator as tdp

tile_matrix = [[3, 1], [2, 2], [2, 3], [2, 4], [3, 3], [3, 4], [4, 4], [4, 5], [5, 5],
    [5, 4], [0, 0], [1, 0], [1, 1], [0, 2], [1, 2], [2, 1], [2, 0], [3, 0],
    [3, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5], [2, 5], [4, 3],
    [4, 0], [5, 0], [4, 1], [5, 1], [4, 2], [5, 2], [5, 3]]
oya_site = [0, 1]
reach_site = [3, 5]
HAI = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    'Tou', 'nan', 'sya', 'pei', 'hak', 'hat', 'cyu' ]
MAX_TURN = 24
TEN_MATRIX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 18, 24]

class DataGenerator:
    @staticmethod
    def data2tiles(_data):
        data = list(_data)
        for i in range(4):
            data[1][i] = sorted(gs.numlist2tiles(data[1][i]))
        for i in range(4):
            for j in range(len(data[2][i])):
                data[2][i][j] = gs.numlist2tiles(data[2][i][j])
        for i in range(4):
            data[3][i] = gs.numlist2tiles(data[3][i])
        '''
        for i in range(len(data[7])):
            for j in range(4):
                data[7][i].pop((3-j)*2)
        '''
        return data
         
    @staticmethod
    def x_y(data):
        return tile_matrix[data][0], tile_matrix[data][1]

    @staticmethod
    def mesen_transfer(_data):
        data = list(_data)
        turn = data[0][0]
        if turn == 'D':
            return data
        hands = data[1]
        mentsu = data[2]
        kawa = data[3]
        if turn == 'E':
            hands.append(hands.pop(0))
            mentsu.append(mentsu.pop(0))
            kawa.append(kawa.pop(0))
            data[5] = (data[5] + 3) % 4
            data[6] = [((x + 3) % 4) for x in data[6]]
            '''
            for i in range(len(data[7])):
                data[7][i].append(data[7][i].pop(0))
            for i in range(len(data[8])):
                data[8][i] = (data[8][i] + 3) % 4
                data[9][i] = (data[9][i] + 3) % 4
            '''
        elif turn == 'F':
            hands.append(hands.pop(0))
            hands.append(hands.pop(0))
            mentsu.append(mentsu.pop(0))
            mentsu.append(mentsu.pop(0))
            kawa.append(kawa.pop(0))
            kawa.append(kawa.pop(0))
            data[5] = (data[5] + 2) % 4
            data[6] = [((x + 2) % 4) for x in data[6]]
            '''
            for i in range(len(data[7])):
                data[7][i].append(data[7][i].pop(0))
                data[7][i].append(data[7][i].pop(0))
            for i in range(len(data[8])):
                data[8][i] = (data[8][i] + 2) % 4
                data[9][i] = (data[9][i] + 2) % 4
            '''
        elif turn == 'G':
            hands = hands.insert(0, hands.pop())
            mentsu = mentsu.insert(0, mentsu.pop())
            kawa = kawa.insert(0, kawa.pop())
            data[5] = (data[5] + 1) % 4
            data[6] = [((x + 1) % 4) for x in data[6]]
            '''
            for i in range(len(data[7])):
                data[7][i].insert(0, data[7][i].pop())
            for i in range(len(data[8])):
                data[8][i] = (data[8][i] + 1) % 4
                data[9][i] = (data[9][i] + 1) % 4
            '''
        
        return data

    @staticmethod
    def discarded_gen(tile):
        tile_matrix = [[0 for i in range(6)] for j in range(6)]
        x, y = DataGenerator.x_y(tile)
        tile_matrix[x][y] = 1
        return tile_matrix

    @staticmethod
    def lp_x_gen(data):
        x = []

        hands = tdp.hands_gen(data[0][0], data[1], data[2], data[5], data[6])
        for hand in hands:
            x.append(hand) #4 layers
        
        kawas = tdp.turns_gen(data[3])
        for i in range(MAX_TURN):
            for j in range(4):
                x.append(kawas[j][i]) #4*24 layers
        for i in range(4):
            x.append(kawas[4][i]) #4 layers

        dora = tdp.dora_gen(data[4])
        x.append(dora) #1 layers
    
        invisual = tdp.invisual_tiles(data)
        x.append(invisual) #1 layers

        discardable = tdp.discardable_gen(data[1], data[2])
        x.append(discardable) #1 layers

        discarded = DataGenerator.discarded_gen(gs.num2tiles(int(data[0][1:])))
        x.append(discarded)
        return x

    @staticmethod
    def lp_y_gen(data, who = 4):
        y = [0 for i in range(len(TEN_MATRIX))]
        ten = 0
        '''
        if who == 4: 
            if data[0][0] == 'D':
                who = 0
            elif data[0][0] == 'E':
                who = 1
            elif data[0][0] == 'F':
                who = 2
            elif data[0][0] == 'G':
                who = 3
        '''     
        for loss_point in data[7]:
            ten += loss_point
        ten = ten / 1000
        if ten == 0:
            ten = 1
        elif ten >= 12 and ten < 16:
            ten = 12
        elif ten >=16 and ten <18:
            ten = 16
        elif ten >=18 and ten <24:
            ten = 18
        elif ten >=24:
            ten = 24
        if ten == 10:
            ten = 9
        y[TEN_MATRIX.index(ten)] = 1
        return y

    @staticmethod
    def lp_data_gen(_data):
        DataGenerator.return_to_last_turn(_data)
        mesen_data = DataGenerator.mesen_transfer(_data)
        mesen_data = DataGenerator.data2tiles(mesen_data)
        return DataGenerator.lp_x_gen(mesen_data), DataGenerator.lp_y_gen(mesen_data)
        '''
        _who = data[8]
        _from_who = data[9]
        mesen_data = []
        if _who[0] == _from_who[0]: #ZIMO
            return False,[], []
            who = int(_who[0])
            if who == 0:
                if data[0][0] != 'D':
                    mesen_data = DataGenerator.mesen_transfer(data)
                    return True, DataGenerator.loss_x_gen(mesen_data), DataGenerator.y_gen(data)
            elif who == 1:
                if data[0][0] != 'E':
                    mesen_data = DataGenerator.mesen_transfer(data)
                    return True, DataGenerator.loss_x_gen(mesen_data), DataGenerator.y_gen(data)
            elif who == 2:
                if data[0][0] != 'F':
                    mesen_data = DataGenerator.mesen_transfer(data)
                    return True, DataGenerator.loss_x_gen(mesen_data), DataGenerator.y_gen(data)
            elif who == 3: 
                if data[0][0] != 'G':
                    mesen_data = DataGenerator.mesen_transfer(data)
                    return True, DataGenerator.loss_x_gen(mesen_data), DataGenerator.y_gen(data)
        else:
            from_who = int(_from_who[0])
            if from_who == 0:
                if data[0][0] == 'D':
                    return True, DataGenerator.lp_x_gen(mesen_data), DataGenerator.lp_y_gen(data)
            elif from_who == 1:
                if data[0][0] == 'E':
                    return True, DataGenerator.lp_x_gen(mesen_data), DataGenerator.lp_y_gen(data)
            elif from_who == 2:
                if data[0][0] == 'F':
                    return True, DataGenerator.lp_x_gen(mesen_data), DataGenerator.lp_y_gen(data)
            elif from_who == 3:
                if data[0][0] == 'G':
                    return True, DataGenerator.lp_x_gen(mesen_data), DataGenerator.lp_y_gen(data)
        return False, [], []
    '''

    @staticmethod
    def return_to_last_turn(data):
        who = data[0][0]
        tile = int(data[0][1:])
        if who == 'D':
            i = 0
        elif who == 'E':
            i = 1
        elif who == 'F':
            i = 2
        elif who == 'G':
            i = 3
        data[1][i].append(data[3][i].pop())

    @staticmethod
    def wt_x_gen(data):
        x = []

        hands = tdp.hands_gen(data[0][0], data[1], data[2], data[5], data[6])
        for hand in hands:
            x.append(hand) #4 layers
        
        kawas = tdp.turns_gen(data[3])
        for i in range(MAX_TURN):
            for j in range(4):
                x.append(kawas[j][i]) #4*24 layers
        for i in range(4):
            x.append(kawas[4][i]) #4 layers

        dora = tdp.dora_gen(data[4])
        x.append(dora) #1 layers
    
        invisual = tdp.invisual_tiles(data)
        x.append(invisual) #1 layers

        discardable = tdp.discardable_gen(data[1], data[2])
        x.append(discardable) #1 layers

        return x

    @staticmethod
    def wt_y_gen(data):
        y = [0 for i in range(34)]
        y[data[10]] = 1
        return y

    @staticmethod
    def wt_data_gen(_data):
        DataGenerator.return_to_last_turn(_data)
        mesen_data = DataGenerator.mesen_transfer(_data)
        mesen_data = DataGenerator.data2tiles(mesen_data)
        return DataGenerator.wt_x_gen(mesen_data), DataGenerator.wt_y_gen(mesen_data)

if __name__ == '__main__':
    testli = [['F30', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50], [49, 23, 30, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3], [117, 130, 41, 73, 79, 77, 68, 59]], [14], 0, [], [['245', '273', '235', '-243', '255', '0', '235', '0']], [0], [1]],
    ['G35', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1, 35]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59]], [14], 0, [], [['245', '273', '235', '-243', '255', '0', '235', '0']], [0], [1]],
    ['D76', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110, 76], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, [], [['245', '273', '235', '-243', '255', '0', '235', '0']], [0], [1]],
    ['E89', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50, 37], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90, 76], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, [0], [['245', '273', '235', '-243', '255', '0', '235', '0']], [0], [1]],
    ['F55', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 58, 50, 37], [49, 23, 22, 61, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [[82, 87, 89]], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90, 76], [116, 100, 131, 126, 135, 0, 2, 65, 34, 89], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, [0], [['245', '273', '235', '-243', '255', '0', '235', '0']], [0], [1]],
    ['D92', [[6, 3, 106, 94, 79, 98, 82, 37, 101, 73, 69, 38, 74, 92], [25, 32, 81, 50, 54, 43, 88, 28, 78, 93, 42, 116, 125], [49, 61, 97, 111, 110, 95, 89, 57, 16, 52, 44, 22, 59], [64, 58, 76, 66, 84, 96, 67, 80, 60, 99]], [[], [], [], [[7, 10, 12]]], [[118, 48, 56, 14, 122, 103, 30, 109, 100], [117, 121, 107, 112, 123, 108, 4, 8, 17, 51], [77, 72, 70, 10, 131, 71, 1, 120, 85, 91], [128, 113, 26, 105, 45, 102, 87, 119, 47, 135]], [33], 1, [], [['123', '-3', '226', '-5', '312', '-3', '339', '11']], [3], [3]],
    ['E42', [[6, 3, 106, 94, 79, 98, 82, 37, 101, 73, 69, 38, 74], [25, 32, 81, 50, 54, 43, 88, 28, 78, 93, 42, 116, 125, 90], [49, 61, 97, 111, 110, 95, 89, 57, 16, 52, 44, 22, 59], [64, 58, 76, 66, 84, 96, 67, 80, 60, 99]], [[], [], [], [[7, 10, 12]]], [[118, 48, 56, 14, 122, 103, 30, 109, 100, 92], [117, 121, 107, 112, 123, 108, 4, 8, 17, 51], [77, 72, 70, 10, 131, 71, 1, 120, 85, 91], [128, 113, 26, 105, 45, 102, 87, 119, 47, 135]], [33], 1, [], [['123', '-3', '226', '-5', '312', '-3', '339', '11']], [3], [3]],
    ['F59', [[6, 3, 106, 94, 79, 98, 82, 37, 101, 73, 69, 38, 74], [25, 32, 81, 50, 54, 43, 88, 28, 78, 93, 116, 125, 90], [61, 97, 111, 110, 95, 89, 57, 16, 52, 22, 59], [64, 58, 76, 66, 84, 96, 67, 80, 60, 99]], [[], [], [[42, 44, 49]], [[7, 10, 12]]], [[118, 48, 56, 14, 122, 103, 30, 109, 100, 92], [117, 121, 107, 112, 123, 108, 4, 8, 17, 51, 42], [77, 72, 70, 10, 131, 71, 1, 120, 85, 91], [128, 113, 26, 105, 45, 102, 87, 119, 47, 135]], [33], 1, [], [['123', '-3', '226', '-5', '312', '-3', '339', '11']], [3], [3]],
    ['G35', [[6, 3, 106, 94, 79, 98, 82, 37, 101, 73, 69, 38, 74], [25, 32, 81, 50, 54, 43, 88, 28, 78, 93, 116, 125, 90], [61, 97, 111, 110, 95, 89, 57, 16, 52, 22], [64, 58, 76, 66, 84, 96, 67, 80, 60, 99, 35]], [[], [], [[42, 44, 49]], [[7, 10, 12]]], [[118, 48, 56, 14, 122, 103, 30, 109, 100, 92], [117, 121, 107, 112, 123, 108, 4, 8, 17, 51, 42], [77, 72, 70, 10, 131, 71, 1, 120, 85, 91, 59], [128, 113, 26, 105, 45, 102, 87, 119, 47, 135]], [33], 1, [], [['123', '-3', '226', '-5', '312', '-3', '339', '11']], [3], [3]],
    ['D94', [[6, 3, 106, 94, 79, 98, 82, 37, 101, 73, 69, 38, 74, 63], [25, 32, 81, 50, 54, 43, 88, 28, 78, 93, 116, 125, 90], [61, 97, 111, 110, 95, 89, 57, 16, 52, 22], [64, 58, 76, 66, 84, 96, 67, 80, 60, 99]], [[], [], [[42, 44, 49]], [[7, 10, 12]]], [[118, 48, 56, 14, 122, 103, 30, 109, 100, 92], [117, 121, 107, 112, 123, 108, 4, 8, 17, 51, 42], [77, 72, 70, 10, 131, 71, 1, 120, 85, 91, 59], [128, 113, 26, 105, 45, 102, 87, 119, 47, 135, 35]], [33], 1, [], [['123', '-3', '226', '-5', '312', '-3', '339', '11']], [3], [3]]]
    #for test in testli:
    with open('../xml_data/fz_test.dat') as f:
        for line in f:
            print line
            gen = gs.data_gen_value(line)
            for test in gen:
                pass
            print test
            DataGenerator.lp_data_gen(test)

