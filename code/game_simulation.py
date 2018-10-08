import csv
import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from data_process import data_process

record_file = '../xml_record.dat'
renamed_dir = '../data/'
shanten = Shanten()

class game_record:
    def __init__(self):
        self.hai0 = 0

    def get_game_record(self, game_tree):
        root = game_tree.getroot()
        record = ''
        for child in root:
            if child.tag == 'INIT':
                record = ''
                record += ET.tostring(child)
                continue
            if record != '':
                if child.tag != 'UN' and child.tag != 'BYE':
                    record += ET.tostring(child)
            if child.tag == 'RYUUKYOKU':
                record = ''
            if child.tag == 'AGARI':
                yield record

    def save_game_record(self, record_dir, save_path):
        drt = os.listdir(record_dir)
        for f in drt:
            file_path = record_dir + f
            game_tree = ET.parse(file_path)
            game_sim = self.get_game_record(game_tree)
            with open(save_path, 'a+') as f:
                for item in game_sim:
                    f.write('<data>' + item + '</data>\n')
            
class GameSimulation:
    @staticmethod
    def init_data(game_tree):
        #get a xml tree
        root = ET.fromstring(game_tree)
        return root

    @staticmethod
    def init_hai(init_tag):
        #get inital hands
        hai = []
        hai0 = init_tag.get('hai0').split(',')
        hai1 = init_tag.get('hai1').split(',')
        hai2 = init_tag.get('hai2').split(',')
        hai3 = init_tag.get('hai3').split(',')
        hai.append([int(s) for s in hai0])
        hai.append([int(s) for s in hai1])
        hai.append([int(s) for s in hai2])
        hai.append([int(s) for s in hai3])
        return hai

    @staticmethod
    def point_change(agari_list):
        final_point = []
        for agari_tag in agari_list:
            final_point.append(int(agari_tag.get('ten').split(',')[1]))
        return final_point
    
    @staticmethod
    def dorahai(init_tag):
        seed = init_tag.get('seed')
        dorahai = seed.split(',')[-1]
        return dorahai

    @staticmethod
    def agari_process(agari_tag):
        who = int(agari_tag.get('who'))
        fromWho = int(agari_tag.get('fromWho'))
        machi = GameSimulation.num2tiles(int(agari_tag.get('machi')))
        return who, fromWho, machi

    @staticmethod
    def data_gen_policy(game_tree):
        '''
            data for policy network
            if there's a reach, generate data from reach to agari
            otherwish, generate data for the final siatus
        '''
        reach = False
        root = GameSimulation.init_data(game_tree)
        mentsu = []
        kawa = []
        point = []
        nagare = []
        who_reach = []
        for child in root:
            if child.tag == 'INIT':
                init_tag = child #element
            elif child.tag == 'AGARI':
                agari_tag = child #element
            else:
                nagare.append(ET.tostring(child)) #string
        hai = GameSimulation.init_hai(init_tag)
        hands = hai
        oya = GameSimulation.init_oya(init_tag)
        dorahai = []
        dorahai.append(GameSimulation.num2tiles(GameSimulation.dorahai(init_tag)))
        who, fromWho, machi = GameSimulation.agari_process(agari_tag)
        for i in range(4):
            kawa.append([])
            mentsu.append([])
        generated = []

        for ngr in nagare:
            tiles = ET.fromstring(ngr)
            if tiles.tag != 'N' and tiles.tag != 'REACH' and tiles.tag != 'DORA':
                _who = tiles.tag[0]
                tile = int(tiles.tag[1:])
                #who get what tile
                if _who == 'T':
                    hands[0].append(tile)
                    yield [tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), who, fromWho, machi]
                elif _who == 'U':
                    hands[1].append(tile)
                    yield [tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), who, fromWho, machi]
                elif _who == 'V':
                    hands[2].append(tile)
                    yield [tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), who, fromWho, machi]
                elif _who == 'W':
                    hands[3].append(tile)
                    yield [tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), who, fromWho, machi]
                #who discard what tile
                elif _who == 'D':
                    hands[0].remove(tile)
                    kawa[0].append(tile)
                elif _who == 'E':
                    hands[1].remove(tile)
                    kawa[1].append(tile)
                elif _who == 'F':
                    hands[2].remove(tile)
                    kawa[2].append(tile)
                elif _who == 'G':
                    hands[3].remove(tile)
                    kawa[3].append(tile)
                    
            elif tiles.tag == 'N':
                _who = tiles.get('who')
                m = tiles.get('m')
                r, h = GameSimulation.m_process(m)
                mentsu[int(_who)].append(h)
                for i in h:
                    if i in hands[int(_who)]:
                        hands[int(_who)].remove(i)
            
            elif tiles.tag == 'DORA':
                dorahai.append(GameSimulation.num2tiles(int(tiles.get('hai'))))

            elif tiles.tag == 'REACH':
                if tiles.get('step') == '2':
                    _who_reach = int(tiles.get('who'))
                    who_reach.append(_who_reach)

    @staticmethod
    def init_oya(init_tag):
        return int(init_tag.get('oya'))

    @staticmethod
    def get_winer(agari_tag):
        who = []
        from_who = []
        for agari in agari_tag:
            who.append(int(agari.get('who')))
            from_who.append(int(agari.get('fromWho')))
        machi = GameSimulation.num2tiles(int(agari.get('machi')))
        return who, from_who, machi

    @staticmethod
    def data_gen_value(game_tree):
        '''
            data for value network
            if there's a reach, generate data from reach to agari
            otherwish, generate data for the final siatus
        '''
        root = GameSimulation.init_data(game_tree)
        mentsu = []
        kawa = []
        point = []
        nagare = []
        agari_tag = []
        who_reach = []
        for child in root:
            if child.tag == 'INIT':
                init_tag = child #element
            elif child.tag == 'AGARI':
                agari_tag.append(child) #element
            else:
                nagare.append(ET.tostring(child)) #string
        hai = GameSimulation.init_hai(init_tag)
        oya = GameSimulation.init_oya(init_tag)
        final_point = GameSimulation.point_change(agari_tag)
        hands = hai
        dorahai = []
        dorahai.append(GameSimulation.num2tiles(GameSimulation.dorahai(init_tag)))
        who, from_who, machi = GameSimulation.get_winer(agari_tag)
        
        for i in range(4):
            kawa.append([])
            mentsu.append([])
        generated = []

        for ngr in nagare:
            tiles = ET.fromstring(ngr)
            if tiles.tag != 'N' and tiles.tag != 'REACH' and tiles.tag != 'DORA':
                _who = tiles.tag[0]
                tile = int(tiles.tag[1:])
                #who get what tile
                if _who == 'T':
                    hands[0].append(tile)
                elif _who == 'U':
                    hands[1].append(tile)
                elif _who == 'V':
                    hands[2].append(tile)
                elif _who == 'W':
                    hands[3].append(tile)
                #who discard what tile
                elif _who == 'D':
                    yield tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), list(final_point), list(who), list(from_who), machi
                    hands[0].remove(tile)
                    kawa[0].append(tile)
                elif _who == 'E':
                    yield tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), list(final_point), list(who), list(from_who), machi
                    hands[1].remove(tile)
                    kawa[1].append(tile)
                elif _who == 'F':
                    yield tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), list(final_point), list(who), list(from_who), machi
                    hands[2].remove(tile)
                    kawa[2].append(tile)
                elif _who == 'G':
                    yield tiles.tag, list(hands), list(mentsu), list(kawa), list(dorahai), oya, list(who_reach), list(final_point), list(who), list(from_who), machi
                    hands[3].remove(tile)
                    kawa[3].append(tile)
                    
            elif tiles.tag == 'N':
                _who = tiles.get('who')
                m = tiles.get('m')
                r, h = GameSimulation.m_process(m)
                mentsu[int(_who)].append(h)
                for i in h:
                    if i in hands[int(_who)]:
                        hands[int(_who)].remove(i)
            
            elif tiles.tag == 'DORA':
                dorahai.append(GameSimulation.num2tiles(int(tiles.get('hai'))))

            elif tiles.tag == 'REACH':
                if tiles.get('step') == '2':
                    _who_reach = int(tiles.get('who'))
                    who_reach.append(_who_reach)

    @staticmethod
    def concatenate_hands_and_mentsu(hands, mentsu):
        for i in range(0, 4):
            if mentsu[i] != []:
                for m in mentsu[i]:
                    hands[i] += m
            hands[i] = list(set(hands[i]))
        return hands

    @staticmethod
    def waiting_tiles(tiles):
        '''
            generate a set of waiting hands from 13 tiles
        '''
        wt = []
        tiles = list(tiles)
        for waiting_tile in range(0, 34):
            try:
                tiles.append(waiting_tile)
                man, pin, sou, honors = GameSimulation.tiles_transform(tiles)
                mjtiles = TilesConverter.string_to_34_array(man=man, pin=pin, sou=sou, honors=honors)
            except:
                continue
            if shanten.calculate_shanten(mjtiles) == -1:
                wt.append(waiting_tile)
            tiles.remove(waiting_tile)
        return wt

    @staticmethod
    def tiles_transform(tiles):
        '''
            transform hands to mpsz for lib-mahjong
        '''
        man = ''
        pin = ''
        sou = ''
        honors = ''
        for tile in tiles:
            if tile < 9:
                man += str(tile + 1)
            elif tile < 18:
                pin += str(tile % 9 + 1)
            elif tile < 27:
                sou += str(tile % 9 + 1)
            else:
                honors += str(tile % 9 + 1)
        return man, pin, sou, honors

    @staticmethod
    def numlist2tiles(tiles):
        hai = []
        for tile in tiles:
            hai.append(GameSimulation.num2tiles(int(tile)))
        return hai

    @staticmethod
    def num2tiles(num):
        num = int(num)
        mpsz = num // 36 #caculate weather a tile is m or p or s or z
        hai = mpsz * 9 + ((num % 36) // 4)
        return hai

    #due with the m to get the mentsu
    @staticmethod
    def m_process(m):
        m = int(m)
        kui = (m & 3)
        r = 0
        t = 0
        h = []
        if (m & (1 << 2)): #SYUNTSU
            t = (m & 0xFC00) >> 10
            r = t % 3 
            t //= 3
            t = (t // 7) * 9 + (t % 7)
            t *= 4
            h = [t + 4 * 0 + ((m & 0x0018) >> 3),
                    t + 4 * 1 + ((m & 0x0060) >> 5),
                    t + 4 * 2 + ((m & 0x0180) >> 7)]
        elif (m & (1 << 3)):#KOUTSU
            unused = (m & 0x0060) >> 5
            t = (m & 0xFE00) >> 9
            r = t % 3
            t = t // 3
            t *= 4
            h = [t, t, t]

            if unused == 0:
                h[0] += 1
                h[1] += 2
                h[2] += 3
            elif unused == 1:
                h[0] += 0
                h[1] += 2
                h[2] += 3
            elif unused == 2:
                h[0] += 0
                h[1] += 1
                h[2] += 3
            elif unused == 3:
                h[0] += 0
                h[1] += 1
                h[2] += 2

        elif (m & (1 << 4)): #CHAKAN
            added = (m & 0x0060) >> 5
            t = (m & 0xFE00) >> 9
            r = t % 3
            t //= 3
            t *= 4
            h = [t, t+1, t+2, t+3]
            
        else:#MINKAN, ANKAN
            hai0 = (m & 0xFF00) >> 8
            if not kui:
                hai0 = (hai0 & ~3) + 3
            t = (hai0 // 4) * 4
            h = [t, t+1, t+2, t+3]
            
        return r, h
            
if __name__ == '__main__':
    '''
    xml_record = game_record()
    xml_record.save_game_record(renamed_dir, record_file)

    hands = [[80, 18, 111, 108, 91, 94, 56, 60, 26, 83, 97, 52, 110], [25, 64, 12, 14, 44, 6, 84, 58, 50, 7, 24, 45, 27], [49, 23, 22, 99, 95, 92, 98, 93, 48, 109], [86, 13, 15, 20, 19, 42, 8, 39, 46, 51, 88, 1, 85]]
    mentsu = [[], [], [[82, 87, 89]], []]
    print hands 
    print GameSimulation.concatenate_hands_and_mentsu(hands, mentsu)
    print hands 
    
    with open('../xml_data/shuf_test.dat', 'r+') as f:
        tests = f.readlines()
    for test in tests:
        gen = GameSimulation.data_gen_policy(test)
    
        for item in gen:
            pass
        if item[7] != item[8]:
            print test
            print item
            #tile = GameSimulation.num2tiles(int(item[0][1:]))
            tile = item[3][item[8]].pop()
            item[1][item[8]].append(tile)
            print item
    '''
    '''
        hands = GameSimulation.concatenate_hands_and_mentsu(item[1], item[2])
        for hai in hands:
                    
            _hai = GameSimulation.numlist2tiles(hai)
            print sorted(_hai)
            #print 'hai', list(sorted(_hai))
            wt = GameSimulation.waiting_tiles(_hai)
            if wt!= []:
                print sorted(_hai), wt

    with open('../test.dat', 'r+') as f:
        test_li = f.readlines()
    line = 0
    for test in test_li:
        gen = GameSimulation.data_gen(test)
        for item in gen:
            hands = GameSimulation.concatenate_hands_and_mentsu(item[1], item[2])
            for hai in hands:
                _hai = GameSimulation.numlist2tiles(hai)
                wt = GameSimulation.waiting_tiles(_hai)
                if wt != []:
                    print item
                    print _hai, wt
    '''
