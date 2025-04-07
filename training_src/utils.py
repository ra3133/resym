import os
import json
import shutil
import time
import glob
import re
import csv
# import openai
from typing import List, Dict
import statistics
import pprint


def init_folder(path, create=True, clean=True, verbose=True):
    # create: whether to create the folder if not exist
    # clean: whether to clean the files in the folder
    if not os.path.exists(path) and create:
        if verbose:
            print('Creating folder '+ str(path))
        os.makedirs(path)
    if clean:
        if verbose:
            print('Cleanning all files recursively in ' + str(path))
        del_file(path)

def del_file(dir_addr, format = None):
    # delete **recursively** all files existing in the folder
    # example format: '*.yaml'
    if format:
        del_files = glob.glob(dir_addr+'/'+ format, recursive=True)
    else:
        del_files = glob.glob(dir_addr+'/*', recursive=True)
    for df in del_files:
        try:
            os.remove(df)
        except:
            pass

def copyfile(src, dst):
    shutil.copyfile(src, dst)

def check_dir_exist(d):
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise argparse.ArgumentTypeError("%s is not a valid work directory" % d)
    return d

def write_csv(path, lines):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for l in lines:
            writer.writerow(l)
            
def get_file_list(dir_addr):

    files = []
    for _,_, filenames in os.walk(dir_addr):
        files.extend(filenames)
        break

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    return files


# def read_file(path):
#     with open(path, "r") as f:
#         ret = f.readlines()
#     return ret


def read_file(path, readlines=True) -> str:
    with open(path, "r") as f:
        if readlines:
            ret = f.readlines()  # -> List[str]
        else:
            ret = f.read()  # -> str
    return ret


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

        
def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_decompiled(fpath) -> List[str]:
    data = read_json(fpath)
    result = []
    for func in data:
        if func.startswith('sub') or func == 'main':
            code = data[func]
            # TODO: confirm heuristics
            if len(code.split('\n')) < 20 and \
                (re.findall('byte_\d+', code) != [] or re.findall('cxa_finalize', code) != []):
                continue
            result.append(data[func])
    return result

def load_original(fpath):
    with open(fpath, 'r') as f:
        return ''.join(f.readlines())
        


# def call_turbo(prompt, history=[], stop_words=None, max_tokens=2048):
#     # key is set in the upper stream function
#     # return response_content, time

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages= history + [
#             {"role": "system", "content": prompt},
#         ],
#         temperature=0,
#         max_tokens=max_tokens,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=stop_words
#     )

#     return response['choices'][0]['message']['content']


# def hex2str(hex_string, reverse=True):
#     # no: hex_string = '0x6E6F65726F706176LL'
#     # yes: hex_string = '6E6F65726F706176'

#     # Remove the '0x' prefix and 'LL' suffix
#     # hex_string = hex_string[2:-2]

#     # Convert the hexadecimal string to bytes
#     byte_array = bytes.fromhex(hex_string)

#     # Convert bytes to string
#     original_string = byte_array.decode()
#     if reverse:
#         return original_string[::-1]
#     else:
#         return original_string

# def str2hex(input_string, reverse=True):
#     # Convert string to bytes
#     if reverse:
#         input_string = input_string[::-1]
#     byte_array = input_string.encode()

#     # Convert bytes to hexadecimal string
#     hex_string = byte_array.hex()

#     return hex_string


def print_stat(data_list: List):
    min_val = min(data_list)
    max_val = max(data_list)
    median = statistics.median(data_list)
    mean = statistics.mean(data_list)
    stdev = statistics.stdev(data_list)
    variance = statistics.variance(data_list)
    print('Len: {}, Min: {}, Max: {}, Median: {}, Average: {}, stdev: {}, Variance: {}'.format(\
        len(data_list), min_val, max_val, median, mean, stdev, variance))

def pprint_dict(d):
    pprint.pprint(d)