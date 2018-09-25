import os
import random
import xml.etree.ElementTree as ET

unprocessed = '../unprocessed_data/'
renamed = '../renamed_data/'
path = '../data/'

#unzip the gzip of mjlog to xml
def unzip_cmd(zip_path, save_path):
    cmd = 'cat '+ zip_path + ' | gzip -d > ' + save_path
    try:
        os.system(cmd)
    except e:
        print e

#rename file
def rename_cmd(pre_name, final_name):
    print pre_name, final_name
    pre_name.replace('&', '\&')
    cmd = 'cp ' + pre_name + ' ' + final_name
    try:
        os.system(cmd)
    except e:
        print e

#rename all the files to prepal for the unzip
def rename_file(data_path):
    drt = os.listdir(data_path)
    games = 0

    for f in drt:
        f = data_path + f
        drts = os.listdir(f)

        for file_path in drts:
            file_path = f + '/' + file_path 
            final_path = renamed + str(games) + '.mjlog'
            os.rename(file_path, final_path)
            games = games + 1

#list all the data file to be processed
def get_data(data_path):
    drt = os.listdir(data_path)
    games = 0

    for f in drt:
        file_path = data_path + f
        save_path = path + str(games) + '.xml'
        unzip_cmd(file_path, save_path)
        games = games + 1
        print file_path, save_path

def generate_valid_dataset(afile):
    line = next(afile)
    for num, aline in enumerate(afile):
        if random.randrange(num+2): continue
        line = aline
    return line


if __name__ == '__main__':
    #unzip('../test.mjlog', '../1.xml')
    #rename_file(unprocessed)
    #get_data(renamed)

    with open('../xml_data/xml_valid_record.dat', 'a+') as val_f:
        for i in range(1000):
            print i
            with open('../xml_data/random_collect_data.dat') as f:
                val_line = generate_valid_dataset(f)
            val_f.write(val_line)
'''
tree = ET.parse(path)
root = tree.getroot()
print root
'''
