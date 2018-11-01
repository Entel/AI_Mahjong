import gzip
import shutil

DAT_PATH = '../data/'
DISCARD_DATA = 'discard_training.dat'
xml_list = [
    'lp_training',
    'lp_validation',
    'wton_training',
    'wton_validation',
    'wt_training',
    'wt_validation',
    ]
dat_list = [
    'discard_training',
    'discard_validation'
    ]
DISCARD_DATA = 'discard_training'

def compress_data(file_path, save_path):
    with open(file_path, 'rb') as f_in, gzip.open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def unzip_data(file_path, save_path):
    with gzip.open(file_path, 'rb') as f_in, open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def compress_list():
    for item in xml_list:
        compress_data('../xml_data/'+ item + '.dat', '../compressed_data/' + item + '.gzip')
        print('Compressed: ' + item)
    for item in dat_list:
        compress_data('../data/'+ item + '.dat', '../compressed_data/' + item + '.gzip')
        print('Compressed: ' + item)

def unzip_list():
    for item in xml_data:
        unzip_data('../compressed_data/'+ item + '.gzip', '../xml_data/' + item + '.dat')
        print('Unzip: ' + item)
    for item in dat_list:
        unzip_data('../compressed_data/'+ item + '.gzip', '../data/' + item + '.dat')
        print('Unzip: ' + item)

'''
large file process
'''
def file_split(datapath, datafile, parts):
    line_count = 0
    p_line_count = 0
    num = 0
    for i in range(parts):
        file_name = datafile + '.p' + str(num)
        open(file_name, 'w').close

    with open(datapath + datafile) as f:
        for line in f:
            line_count += 1

    with open(datapath + datafile) as f:
        for line in f:
            if p_line_count <= round((line_count + 4) / 4):
                file_name = datafile + '.p' + str(num)
                with open(datapath + file_name, 'a') as pf:
                    p_line_count += 1
                    pf.write(line)
            else:
                num += 1
                p_line_count = 0
                file_name = datafile + '.p' + str(num)
                with open(datapath + file_name, 'a') as pf:
                    p_line_count += 1
                    pf.write(line)

def file_concatenate(datapath, datafile, parts):
    with open(datapath + datafile, 'a') as f:
        for i in range(parts):
            file_name = datafile + '.p' + str(i)
            with open(datapath + file_name) as pf:
                for line in pf:
                    f.write(line)
    
def compress_large_file(datapath, fname, parts):
    file_split(datapath, fname, parts)
    for i in range(parts):
        compress_data(datapath+ fname + '.dat.p' + str(i), '../compressed_data/' + fname + '.gzip.p' + str(i))
        print('Compressed: ' + fname + ' part' + str(i))

def unzip_large_file(datapath, fname, parts):
    for i in range(parts):
        unzip_data('../compressed_data/' + fname + '.gzip.p' + str(i), datapath + fname + '.dat.p' + str(i))
        print('Unzip: ' + fname + ' part' + str(i))
    file_concatenate(datapath, fname, parts)

if __name__ == '__main__':
    #compress_list()
    #compress_large_file(DAT_PATH, DISCARD_DATA, 4)
    #unzip_list()
    unzip_data('../compressed_data/discard_validation.gzip', '../data/discard_validation.dat')
