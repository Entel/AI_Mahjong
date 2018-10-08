import xml.etree.ElementTree as ET
import pandas as pd
import csv
import os

embedding_file = '../embedding.csv'
final_hai_file = '../final_hai.csv'
DATA = '../data/'

class data_process:
    
    def __init__(self, file_path, save_path):
        self.file_path = file_path
        self.save_path = save_path

    #collect the hai of the winner
    def get_agari(self, filename):
        try:
            self.tree = ET.parse(filename)
            self.root = self.tree.getroot()

            return True, self.root.iter('AGARI')
        except:
            return False, ''
        #for self.agari_tag in self.root.iter('AGARI'):
        #    print self.agari_tag.attrib

    #due with the m to get the mentsu
    def mentsu_process(self, m):
        m = int(m)
        self.kui = (m & 3)
        self.r = 0
        self.h = []
        if (m & (1 << 2)): #SYUNTSU
            self.t = (m & 0xFC00) >> 10
            self.r = self.t % 3 #who
            self.t /= 3
            self.t = (self.t / 7) * 9 + (self.t % 7)
            self.t *= 4
            self.h = [self.t + 4 * 0 + ((m & 0x0018) >> 3),
                    self.t + 4 * 1 + ((m & 0x0060) >> 5),
                    self.t + 4 * 2 + ((m & 0x0180) >> 7)]
        elif (m & (1 << 3)):#KOUTSU
            self.unused = (m & 0x0060) >> 5
            self.t = (m & 0xFE00) >> 9
            self.r = self.t % 3
            self.t = self.t / 3
            self.t *= 4
            self.h = [self.t, self.t, self.t]

            if self.unused == 0:
                self.h[0] += 1
                self.h[1] += 2
                self.h[2] += 3
            elif self.unused == 1:
                self.h[0] += 0
                self.h[1] += 2
                self.h[2] += 3
            elif self.unused == 2:
                self.h[0] += 0
                self.h[1] += 1
                self.h[2] += 3
            elif self.unused == 3:
                self.h[0] += 0
                self.h[1] += 1
                self.h[2] += 2

        elif (m & (1 << 4)): #CHAKAN
            self.added = (m & 0x0060) >> 5
            self.t = (m & 0xFE00) >> 9
            self.r = self.t % 3
            self.t /= 3
            self.t *= 4
            self.h = [self.t, self.t+1, self.t+2, self.t+3]
            
        else:#MINKAN, ANKAN
            self.hai0 = (m & 0xFF00) >> 8
            if not self.kui:
                self.hai0 = (self.hai0 & ~3) + 3
            self.t = (self.hai0 / 4) * 4
            self.h = [self.t, self.t+1, self.t+2, self.t+3]
            
        return self.r, self.h

    #save the data to a csv file
    def add_to_csv(self, path, li):
        with open(path, 'a+') as csv_file:
            wr = csv.writer(csv_file, dialect = 'excel')
            wr.writerow(li)

    #return a list of data file
    def list_path_of_data(self):
        drts = os.listdir(self.file_path)
        for drt in drts:
            yield self.file_path + drt
    
    #generate the data of final hai for training the embedding map
    def gen_embedding_data(self, file_name):
        ner, agari = self.get_agari(file_name)
        if ner:
            for agari_tag in agari:
                _hai = agari_tag.attrib['hai']
                hai = _hai.split(',')
                final_hai = hai
        
                if 'm' in agari_tag.attrib:
                    _m = agari_tag.attrib['m']
                    m = _m.split(',')
                    for mentsu in m:
                        _mentsu = self.mentsu_process(mentsu)
                        final_hai = _mentsu[1] + final_hai
                yield final_hai

    #transfer the number of tiles to tiles
    def num2tiles(self, num):
        num = int(num)
        self.mpsz = num / 36 #caculate weather a tile is m or p or s or z
        self.hai = self.mpsz * 9 + ((num % 36) / 4)
        return self.hai

    #read csv
    def read_csv(self, filename):
        with open(filename, 'rb') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                yield row

if __name__ == '__main__':
    process = data_process(DATA, '1')
    '''
    agari = process.get_agari('../1.xml')
    for agari_tag in agari:
        _hai = agari_tag.attrib['hai']
        hai = _hai.split(',')
        
        who = agari_tag.attrib['who']
        from_who = agari_tag.attrib['fromWho']
        dora = agari_tag.attrib['doraHai']
        ten = agari_tag.attrib['ten'].split(',').[1]
        
        final_hai = hai
        
        if 'm' in agari_tag.attrib:
            _m = agari_tag.attrib['m']
            m = _m.split(',')
            for mentsu in m:
                _mentsu = process.mentsu_process(mentsu)
                final_hai += _mentsu[1]
        
        if 'doraHaiUra' in agari_tag.attrib:
            _ura = agari_tag.attrib['doraHaiUra']
            ura = _ura.split(',')
    ''' 
    
    ''' 
    data_path = process.list_path_of_data()
    for data_file in data_path:
        print data_file
        agari_data = process.gen_embedding_data(data_file)
        for final_hai in agari_data:
            process.add_to_csv(embedding_file, final_hai)
            print final_hai

    hai_list = process.read_csv(embedding_file)
    for hai in hai_list:
        tiles = []
        for tile in hai:
            tiles.append(process.num2tiles(tile))
        process.add_to_csv(final_hai_file, tiles)
        print tiles
    print process.mentsu_process(44107)
    print process.mentsu_process(42603)
    print process.mentsu_process(71)
    print process.mentsu_process(34935)
    print process.mentsu_process(61695)
    ''' 
