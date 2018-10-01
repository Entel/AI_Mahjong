import numpy as np
import xml.etree.ElementTree as ET
from game_simulation import GameSimulation as gs

'''
To add up the data into balance for training
'''

XML_DATA = '../xml_data/shuf_xml_record.dat'
ZIMO_DATA = '../xml_data/zimo.dat'
HOZYU_DATA = '../xml_data/hozyu.dat'
LP_XML_DATA = '../xml_data/lp_rd_collect_data.dat'
LP_TRAINING = '../xml_data/lp_training.dat'
LP_VALIDATION = '../xml_data/lp_validation.dat'
WT_XML_DATA = '../xml_data/wt_rd_collect_data.dat'
WT_TRAINING = '../xml_data/wt_training.dat'
WT_VALIDATION = '../xml_data/wt_validation.dat'
TEN_MATRIX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 18, 24]

class ResultStatistics():
    @staticmethod
    def init_data(game_tree):
        root = ET.fromstring(game_tree)
        return root 

    @staticmethod
    def ten_range(agari_tag):
        ten = int(agari_tag.get('ten').split(',')[1])
        ten = ten / 1000
        if ten == 0:
            ten = 1
        if ten >= 12 and ten < 16:
            ten = 12
        elif ten >=16 and ten <18:
            ten = 16
        elif ten >=18 and ten <24:
            ten = 18
        elif ten >=24:
            ten = 24
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
    def waiting_tile(agari_tag):
        machi = int(agari_tag.get('machi'))
        tile = ResultStatistics.num2tiles(machi)
        return tile

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
    def wton_add_up(datapath):
        st = [0 for i in range(4)]
        with open(datapath, 'r') as f:
            for line in f:
                root = ResultStatistics.init_data(line)
                zimo, agaris, who = ResultStatistics.agari_tag(root)
                for _who in who:
                    st[_who] += 1
        return st

    @staticmethod
    def num2tiles(num):
        num = int(num)
        mpsz = num / 36 #caculate weather a tile is m or p or s or z
        hai = mpsz * 9 + ((num % 36) / 4)
        return hai

    @staticmethod
    def lp_collect_data():
        st = [1000 for i in range(len(TEN_MATRIX))]
        for i in range(2):
            with open(XML_DATA, 'r') as f:
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
                            print st

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
                        print st
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
                    

if __name__ == '__main__':
    #print ResultStatistics.lp_collect_data()
    #print ResultStatistics.lp_add_up('../xml_data/1.xml')
    #print ResultStatistics.add_up('../xml_data/xml_valid_record.dat')
    #print ResultStatistics.machi_add_up('../xml_data/shuf_xml_record.dat')
    #print ResultStatistics.wt_collect_data()
    #print ResultStatistics.wt_add_up(WT_XML_DATA)
    #ResultStatistics.file_division(11200, LP_XML_DATA, LP_TRAINING, LP_VALIDATION)
    print ResultStatistics.lp_add_up(LP_TRAINING)
    print ResultStatistics.lp_add_up(LP_VALIDATION)
    #print ResultStatistics.wton_add_up(XML_DATA)
    #ResultStatistics.agari_type_classification()
