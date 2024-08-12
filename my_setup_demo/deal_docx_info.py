__version__ = "0.1.1"



import os
import json

def save2json(data, json_path: str):
    '''
    保存为json格式
    '''

    with open(json_path, mode='wt', encoding='utf8') as f:
        print('datas is {}'.format(len(data)))
        json.dump(data, f, ensure_ascii=False, indent=4)
    return os.path.abspath(json_path)


def read_json(json_path: str):
    '''
    读取json文件问python对象
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
