import os
import numpy as np
from game_simulation import GameSimulation

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

class DataGenerator:
    @staticmethod
    def data2tiles(_data):
        data = list(_data)
        for i in range(4):
            data[1][i] = sorted(GameSimulation.numlist2tiles(data[1][i]))
        for i in range(4):
            for j in range(len(data[2][i])):
                data[2][i][j] = GameSimulation.numlist2tiles(data[2][i][j])
        for i in range(4):
            data[3][i] = GameSimulation.numlist2tiles(data[3][i])
        for i in range(len(data[4])):
            data[4][i] = GameSimulation.num2tiles(int(data[4][i]))
        return data

    @staticmethod
    def x_y(data):    
        return tile_matrix[data][0], tile_matrix[data][1]

    @staticmethod
    def mesen_transfer(_data):
        data = list(_data)
        turn = data[0][0]
        if turn == 'T':
            return data
        hands = data[1]
        mentsu = data[2]
        kawa = data[3]
        if turn == 'U':
            hands.append(hands.pop(0))
            mentsu.append(mentsu.pop(0))
            kawa.append(kawa.pop(0))
            data[5] = (data[5] + 3) % 4
            data[6] = [((x + 3) % 4) for x in data[6]]
        elif turn == 'V':
            hands.append(hands.pop(0))
            hands.append(hands.pop(0))
            mentsu.append(mentsu.pop(0))
            mentsu.append(mentsu.pop(0))
            kawa.append(kawa.pop(0))
            kawa.append(kawa.pop(0))
            data[5] = (data[5] + 2) % 4
            data[6] = [((x + 2) % 4) for x in data[6]]
        elif turn == 'W':
            hands = hands.insert(0, hands.pop())
            mentsu = mentsu.insert(0, mentsu.pop())
            kawa = kawa.insert(0, kawa.pop())
            data[5] = (data[5] + 1) % 4
            data[6] = [((x + 1) % 4) for x in data[6]]
        
        return data

    @staticmethod
    def hands_gen(turn, data, mentsu, oya, reach):
        '''
        genertate hands from the table status
        own: hands
        oppenent: mentsu
        '''
        hands = [[[0 for i in range(6)] for j in range(6)] for k in range(4)]
        tile = data[0]
        for item in tile:
            x, y = DataGenerator.x_y(item)
            hands[0][x][y] += 0.25
        for i in range(1, 4):
            if mentsu[i] != []:
                for item in mentsu[i]:
                    for item2 in item:
                        x, y = DataGenerator.x_y(item2)
                        hands[i][x][y] += 0.25
        hands[oya][oya_site[0]][oya_site[1]] = 1
        if reach != []:
            for item in reach:
                hands[item][reach_site[0]][reach_site[1]] = 1
        return hands

    @staticmethod
    def turns_gen(kawa):
        '''
        generate a list of discarded tiles in every turn(max_turn*4*6*6) and a list of all tiles discarded(4*6*6)
        '''
        turns_0 = [[[0 for i in range(6)] for j in range(6)] for k in range(MAX_TURN)]
        turns_1 = [[[0 for i in range(6)] for j in range(6)] for k in range(MAX_TURN)]
        turns_2 = [[[0 for i in range(6)] for j in range(6)] for k in range(MAX_TURN)]
        turns_3 = [[[0 for i in range(6)] for j in range(6)] for k in range(MAX_TURN)]
        kawa_matrix = [[[0 for i in range(6)] for j in range(6)] for k in range(4)]
        for i in range(len(kawa[0])):
            x, y = DataGenerator.x_y(kawa[0][i])
            turns_0[i][x][y] = 1
        for i in range(len(kawa[1])):
            x, y = DataGenerator.x_y(kawa[1][i])
            turns_1[i][x][y] = 1
        for i in range(len(kawa[2])):
            x, y = DataGenerator.x_y(kawa[2][i])
            turns_2[i][x][y] = 1
        for i in range(len(kawa[3])):
            x, y = DataGenerator.x_y(kawa[3][i])
            turns_3[i][x][y] = 1
        
        for i in range(4):
            for tile in kawa[i]:
                x, y = DataGenerator.x_y(tile)
                kawa_matrix[i][x][y] += 0.25

        return turns_0, turns_1, turns_2, turns_3, kawa_matrix

    @staticmethod
    def dora_gen(doras):
        dora_matrix = [[0 for i in range(6)] for j in range(6)]
        for dora in doras:
            x, y = DataGenerator.x_y(dora)
            dora_matrix[x][y] = 1
        return dora_matrix

    @staticmethod
    def discardable_gen(hands, mentsus):
        '''
        genertate the tile that can discard
        '''
        dis_matrix = [[0 for i in range(6)] for j in range(6)]
        hand = list(hands[0])
        if mentsus[0] != []:
            for mentsu in mentsus[0]:
                for item in mentsu:
                    if item in hand:
                        hand.remove(item)
        for item in hand:
            x, y = DataGenerator.x_y(item)
            dis_matrix[x][y] += 0.25
        return dis_matrix

    @staticmethod
    def invisual_tiles(data):
        inv_matrix = [[1 for i in range(6)] for j in range(6)]
        for tile in data[1][0]:
            x, y = DataGenerator.x_y(tile)
            inv_matrix[x][y] -= 0.25
        for i in range(1, 4):
            if data[2][i] != []:
                for item in data[2][i]:
                    for tile in item:
                        x, y = DataGenerator.x_y(tile)
                        inv_matrix[x][y] -= 0.25
        for kawa in data[3]:
            for tile in kawa:
                x, y = DataGenerator.x_y(tile)
                if inv_matrix[x][y] > 0:
                    inv_matrix[x][y] -= 0.25
        return inv_matrix

    @staticmethod
    def train_x_gen(data):
        x = []
        
        hands = DataGenerator.hands_gen(data[0][0], data[1], data[2], data[5], data[6])
        for hand in hands:
            x.append(hand) #4 layers
        
        kawas = DataGenerator.turns_gen(data[3])
        for i in range(MAX_TURN):
            for j in range(4):
                x.append(kawas[j][i]) #4*24 layers
        for i in range(4):
            x.append(kawas[4][i]) #4 layers

        dora = DataGenerator.dora_gen(data[4])
        x.append(dora) #1 layers
    
        invisual = DataGenerator.invisual_tiles(data)
        x.append(invisual) #1 layers

        discardable = DataGenerator.discardable_gen(data[1], data[2])
        x.append(discardable) #1 layers

        return x

    @staticmethod
    def data_preprocess(_data):
        data = DataGenerator.data2tiles(_data)
        data = DataGenerator.mesen_transfer(data)
        return data

    @staticmethod
    def generate_data_for_waiting(_data):
        data = DataGenerator.data_preprocess(_data)
        wt = []
        wton = [0, 0, 0, 0]
        waiting = False
        for hand in data[1]:
            wt.append(gs.waiting_tiles(hand))
        for i in range(1, 4):
            if wt[i] != []:
                wton[i] = 1
                waiting = True
        if waiting:
            x = dg.train_x_gen(data)
            return waiting, x, wt, wton
        else:
            return waiting, [], [], []

    @staticmethod
    def tiles_matrix_gen(tiles):
        tiles_matrix = [[[0 for i in range(6)] for j in range(6)] for k in range(4)]
        for i in range(4):
            if tiles[i] != []:
                for tile in tiles[i]:
                    x, y = dg.x_y(tile)
                    tiles_matrix[i][x][y] = 1
        return tiles_matrix

if __name__ == '__main__':
    testli = [['W35', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1, 35]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59]], [14], 0, []],
['T76', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110, 76], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, []],
['U37', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 89, 58, 50, 37], [49, 23, 87, 22, 61, 82, 99, 95, 53, 92, 55, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90, 76], [116, 100, 131, 126, 135, 0, 2, 65, 34], [38, 121, 125, 5, 75, 74, 70, 3, 30], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, [0]],
['W103', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 58, 50, 37], [49, 23, 22, 61, 99, 95, 53, 92, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1, 103]], [[], [], [[82, 87, 89]], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90, 76], [116, 100, 131, 126, 135, 0, 2, 65, 34, 89], [38, 121, 125, 5, 75, 74, 70, 3, 30, 55], [117, 130, 41, 73, 79, 77, 68, 59, 35]], [14], 0, [0]],
['T62', [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110, 62], [25, 64, 12, 14, 44, 36, 6, 40, 84, 21, 58, 50, 37], [49, 23, 22, 61, 99, 95, 53, 92, 98, 93], [86, 29, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1]], [[], [], [[82, 87, 89]], []], [[31, 122, 128, 102, 106, 127, 11, 54, 90, 76], [116, 100, 131, 126, 135, 0, 2, 65, 34, 89], [38, 121, 125, 5, 75, 74, 70, 3, 30, 55], [117, 130, 41, 73, 79, 77, 68, 59, 35, 103]], [14], 0, [0]]]
    for test in testli:
        test = DataGenerator.data_preprocess(test)
        wt = []
        wton = [0, 0, 0, 0]
        for hand in test[1]:
            wt.append(GameSimulation.waiting_tiles(hand))
        for i in range(4):
            if wt[i] != []:
                wton[i] = 1
        x = DataGenerator.train_x_gen(test)
        print x
        print wt, wton
        print np.array(x).shape

        '''
        data = DataGenerator.data2tiles(test)
        print data
        data = DataGenerator.mesen_transfer(data)
        print data
        print DataGenerator.hands_gen(data[0][0], data[1], data[2])
        #print DataGenerator.turns_gen(data[3])
        print DataGenerator.dora_gen(data[4])
        print DataGenerator.discardable_gen(data[1], data[2])
        print DataGenerator.invisual_tiles(data)
        '''
    '''
    #show matrix
    m = [[-1 for x in range(6)] for y in range(6)]
    for i in range(0, 34):
        x = tile_matrix[i][0]
        y = tile_matrix[i][1]
        
        m[x][y] = HAI[i]
    print m
    '''
