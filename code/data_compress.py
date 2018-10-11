import gzip
import shutil

filelist = [
    'lp_training',
    'lp_validation',
    'wton_training',
    'wton_validation',
    'wt_training',
    'wt_validation',
    ]

def compress_data(file_path, save_path):
    with open(file_path, 'rb') as f_in, gzip.open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def unzip_data(file_path, save_path):
    with gzip.open(file_path, 'rb') as f_in, open(save_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

if __name__ == '__main__':
    for item in filelist:
        compress_data('../xml_data/'+ item + '.dat', '../compressed_data/' + item + '.gzip')
        print('Compressed: ' + item)
