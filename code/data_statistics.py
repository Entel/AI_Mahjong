import numpy as np
import csv
import xml.etree.ElementTree as ET
from game_simulation import GameSimulation as gs

'''
To add up the data into balance for training
'''

XML_DATA = '../xml_data/shuf_xml_record.dat'
ZIMO_DATA = '../xml_data/zimo.dat'
HOZYU_DATA = '../xml_data/hozyu.dat'
LP_XML_DATA = '../xml_data/lp_xml_data.dat'
LP_TRAINING = '../xml_data/lp_training.dat'
LP_VALIDATION = '../xml_data/lp_validation.dat'
WT_XML_DATA = '../xml_data/wt_rd_collect_data.dat'
WT_TRAINING = '../xml_data/wt_training.dat'
WT_VALIDATION = '../xml_data/wt_validation.dat'
HOZYU_TRAINING = '../xml_data/wton_training.dat'
HOZYU_VALIDATION = '../xml_data/wton_validation.dat'
TMP_FILE = '../data/tmp.dat'
DISCARD_DATA = '../data/discard.dat'
DISCARD_TRAINING = '../data/discard_training.dat'
DISCARD_VALIDATION = '../data/discard_validation.dat'

TEN_MATRIX = [1, 2, 3, 4, 5, 6, 8, 12, 18]

class ResultStatistics():
    @staticmethod
    def init_data(game_tree):
        root = ET.fromstring(game_tree)
        return root 

    @staticmethod
    def ten_range(agari_tag):
        ten = int(agari_tag.get('ten').split(',')[1])
        ten = int(round(ten / 1000))
        if ten == 0:
            ten = 1
        if ten >= 7 and ten < 8:
            ten = 8
        elif ten >= 9 and ten < 12:
            ten = 12
        elif ten > 12:
            ten = 18
        return TEN_MATRIX.index(ten)

    @staticmethod
    def agari_tag(root):
        agaris = []
        who = []
        zimo =  True
        for agari in root.iter('AGARI'):
            who.append(int(agari.get('who')))
            fromWho = agari.get('fromWho')
            if who[0] != int(fromWho):
                zimo = False
            agaris.append(agari)
        return zimo, agaris, who

    @staticmethod
    def agari_type_classification():
        with open(XML_DATA) as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, agaris, who = ResultStatistics.agari_tag(root)
                if zimo:
                    with open(ZIMO_DATA, 'a+') as zdf:
                        zdf.write(line)
                else:
                    with open(HOZYU_DATA, 'a+') as hdf:
                        hdf.write(line)

    @staticmethod
    def zimo_add_up(datapath):
        st = [0, 0]
        with open(datapath, 'r') as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, agaris, who = ResultStatistics.agari_tag(root)
                if zimo:
                    st[1] += 1
                else:
                    st[0] += 1
        return st
    @staticmethod

    def lp_add_up(datapath):
        st = [0 for i in range(len(TEN_MATRIX))]
        with open(datapath, 'r') as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, agaris, who = ResultStatistics.agari_tag(root)
                if not zimo:
                    for agari in agaris:
                        st[ResultStatistics.ten_range(agari)] += 1
        return st
                

    @staticmethod
    def num2tiles(num):
        num = int(num)
        mpsz = num // 36 #caculate weather a tile is m or p or s or z
        hai = mpsz * 9 + ((num % 36) // 4)
        return hai

    @staticmethod
    def lp_collect_data():
        st = [5000 for i in range(len(TEN_MATRIX))]
        for i in range(2):
            with open(HOZYU_DATA, 'r') as f:
                #read a line of data and turn it into a game tree
                for line in f:
                    root = ResultStatistics.init_data(line)
                    agari = root.find('AGARI')
                    #judge if it's zimo
                    who = agari.get('who')
                    fromWho = agari.get('fromWho')
                    if who != fromWho: 
                        #if it is needed
                        ten_index = ResultStatistics.ten_range(agari)
                        if st[ten_index] != 0:
                            #add to new file
                            with open(LP_XML_DATA, 'a+') as rf:
                                rf.write(line)
                            st[ten_index] -= 1

        return st

    @staticmethod
    def waiting_tile(agari_tag):
        machi = int(agari_tag.get('machi'))
        tile = ResultStatistics.num2tiles(machi)
        return int(tile)

    @staticmethod
    def wt_add_up(datapath):
        st = [0 for i in range(34)]
        with open(datapath, 'r') as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, agaris, who = ResultStatistics.agari_tag(root)
                for agari in agaris:
                    st[ResultStatistics.waiting_tile(agari)] += 1
        return st

    @staticmethod
    def wt_collect_data():
        st = [4000 for i in range(34)]
        for j in range(2):
            with open(XML_DATA, 'r') as f:
                #read a line of data and turn it into a game tree
                for line in f:
                    root = ResultStatistics.init_data(line)
                    agari = root.find('AGARI')
                    machi = ResultStatistics.num2tiles(agari.get('machi'))
                    if st[machi] != 0:
                        #add to new file
                        with open(WT_XML_DATA, 'a+') as rf:
                            rf.write(line)
                        st[machi] -= 1
        return st

    @staticmethod
    def file_division(line_count, origin_data, training_data, validation_data):
        with open(origin_data) as f:
            for line in f:
                if line_count != 0:
                    line_count -= 1
                    with open(training_data, 'a+') as tdf:
                        tdf.write(line)
                else:
                    with open(validation_data, 'a+') as vdf:
                        vdf.write(line)
                    
    @staticmethod
    def wton_agari_process(root):    
        agaris = []
        whos = []
        zimo =  True
        for agari in root.iter('AGARI'):
            who = int(agari.get('who'))
            whos.append(who)
            fromWho = agari.get('fromWho')
            if whos[0] != int(fromWho):
                zimo = False
        return zimo, int(fromWho), whos

    @staticmethod
    def wton_add_up(datapath):
        st = [0 for i in range(4)]
        with open(datapath) as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, fromWho, who = ResultStatistics.wton_agari_process(root)
                if fromWho == 0:
                    for _who in who:
                        st[_who] += 1
                elif fromWho == 1:
                    for _who in who:
                        st[(_who + 3) % 4] += 1
                elif fromWho == 2:
                    for _who in who:
                        st[(_who + 2) % 4] += 1
                elif fromWho == 3:
                    for _who in who:
                        st[(_who + 1) % 4] += 1
        return st

    @staticmethod
    def list_to_str(li):
        li[0] = str(li[0])

        for i in range(len(li[1])):
            item = ','.join(str(e) for e in li[1][i])
            li[1][i] = item
        li[1] = '|'.join(li[1])

        for i in range(len(li[2])):
            if not li[2][i]:
                li[2][i] = '999'
            else:
                _item = []
                for j in range(len(li[2][i])):
                    if j != 0:
                        li[2][i][0] += li[2][i][j]
                li[2][i] = li[2][i][0]
                li[2][i] = ','.join(str(e) for e in li[2][i])
        li[2] = '|'.join(li[2])

        for i in range(len(li[3])):
            if not li[3][i]:
                li[3][i] = '999'
            else:
                item = ','.join(str(e) for e in li[3][i])
                li[3][i] = item
        li[3] = '|'.join(li[3])

        li[4] = ','.join(str(e) for e in li[4])
        li[5] = str(li[5])
        if not li[6]:
            li[6] = '999'
        else:
            li[6] = ','.join(str(e) for e in li[6])
        li[7] = ','.join(str(e) for e in li[7])
        li[8] = ','.join(str(e) for e in li[8])
        li[9] = ','.join(str(e) for e in li[9])
        li[10] = str(li[10])
        return '||'.join(li)

    @staticmethod
    def discard_add_up(datapath):
        st = [0 for i in range(34)]
        with open(datapath) as f:
            for line in f:
                try:
                    game = gs.data_gen(line)
                    who = game[0][8][0]
                    if who == 0:
                        for item in game:
                            if item[0][0] == 'D':
                                tile = item[0][1:]
                                #write a line into tmp file
                                with open(TMP_FILE, 'a') as tff:
                                    status_str = ResultStatistics.list_to_str(item) + '\n'
                                    tff.write(status_str)
                                st[ResultStatistics.num2tiles(tile)] += 1
                    elif who == 1:
                        for item in game:
                            if item[0][0] == 'E':
                                tile = item[0][1:]
                                #write a line into tmp file
                                with open(TMP_FILE, 'a') as tff:
                                    status_str = ResultStatistics.list_to_str(item) + '\n'
                                    tff.write(status_str)
                                st[ResultStatistics.num2tiles(tile)] += 1
                    elif who == 2:
                        for item in game:
                            if item[0][0] == 'F':
                                tile = item[0][1:]
                                #write a line into tmp file
                                with open(TMP_FILE, 'a') as tff:
                                    status_str = ResultStatistics.list_to_str(item) + '\n'
                                    tff.write(status_str)
                                st[ResultStatistics.num2tiles(tile)] += 1
                    elif who == 3:
                        for item in game:
                            if item[0][0] == 'G':
                                tile = item[0][1:]
                                #write a line into tmp file
                                with open(TMP_FILE, 'a') as tff:
                                    status_str = ResultStatistics.list_to_str(item) + '\n'
                                    tff.write(status_str)
                                st[ResultStatistics.num2tiles(tile)] += 1
                except:
                    continue
        return st

    @staticmethod
    def discard_collect(datapath, savepath):
        st = [100000 for i in range(34)]
        with open(TMP_FILE) as f:
            with open(DISCARD_DATA, 'a+') as df:
                for line in f:
                    li = line.split('||') 
                    tile = ResultStatistics.num2tiles(li[0][1:])
                    if st[tile] != 0:
                        df.write(line)
                        st[tile] -= 1
        return st

    @staticmethod
    def data_combine(origin_data, addition_data):
        with open(addition_data, 'r') as af, open(origin_data) as of:
            for line in af:
                of.write(line)
        

if __name__ == '__main__':
    #print(ResultStatistics.lp_collect_data())
    #print(ResultStatistics.lp_add_up(HOZYU_DATA))
    #print ResultStatistics.zimo_add_up(XML_DATA)
    #print ResultStatistics.machi_add_up('../xml_data/shuf_xml_record.dat')
    #print ResultStatistics.wt_collect_data()
    #print(ResultStatistics.wt_add_up(HOZYU_DATA))
    #ResultStatistics.file_division(4000*9, LP_XML_DATA, LP_TRAINING, LP_VALIDATION)
    #print(ResultStatistics.lp_add_up(LP_TRAINING))
    #print(ResultStatistics.lp_add_up(LP_VALIDATION))
    #print(ResultStatistics.wton_add_up(HOZYU_DATA))
    #ResultStatistics.agari_type_classification()
    #ResultStatistics.file_division(80000*3, HOZYU_DATA, HOZYU_TRAINING, HOZYU_VALIDATION)
    #print(ResultStatistics.wton_add_up(HOZYU_TRAINING))
    #print(ResultStatistics.wton_add_up(HOZYU_VALIDATION))

    #print(ResultStatistics.discard_add_up(XML_DATA))
    #print(ResultStatistics.discard_collect(TMP_FILE, DISCARD_DATA))
    #ResultStatistics.file_division(99000*34, DISCARD_DATA, DISCARD_TRAINING, DISCARD_VALIDATION)
