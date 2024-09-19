'''
解析文档
'''
import json
import os
import re

import pandas as pd


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


def read_excel(new_json_path, xlsx_path):
    """
    解析文档
    """
    df = pd.read_excel(xlsx_path)
    data2 = json.loads(df.to_json(orient='records'))
    new_data = []
    for row in data2:
        content = row['内容']
        if content is None or not content:
            continue
        content = re.sub(r'.*docx', '', content)
        content = re.sub(r'.*doc', '', content)
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'　+\n', '\n', content)
        content = re.sub(r'\n+', '\n', content)
        # 结尾换行处理
        content = re.sub(r'$\n', '', content)
        if len(content) < 99:
            # 不要小于 99 字的文本
            continue
        new_data.append({
            'text': content
        })
    new_json_path = save2json(new_data, new_json_path)
    return new_json_path


if __name__ == '__main__':
    new_json_path = read_excel('/home/zj/LLaMA-Factory/data',
                               '/tmp/gradio/7a22e4d94f829041d7549650203362f91ee765f1/医疗器械-政策法规.xlsx')
    print('new_json_path is {}'.format(new_json_path))
