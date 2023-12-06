# coding = utf-8
import imp
import os
import sys
import json
from os.path import join

# 读取文件的每一行, 返回列表
def get_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        for s in f.readlines():
            s = s.strip()
            if s:
                lines.append(s)
        return lines


def load_json_file(filename):
    """
    :param filename: 文件名
    :return: 数据对象，json/list
    """
    with open(filename, encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_file(filename):
    """
    :param filename: 文件名
    :return: 数据对象，json/list
    """
    data = []
    for line in get_lines(filename):
        js = json.loads(line)
        data.append(js)
    return data


# 把content列表保存成文本文件
def save_file(filename, content):
    """
    :param filename: 输出文件名
    :param content: 句子列表 默认每个元素自带换行啊
    :return:
    """
    if len(content) == 0:
        print('no content')
        return
    one = content[0]
    has_new_line = False
    if one.endswith('\n'):
        has_new_line = True
    
    with open(filename, 'w', encoding='utf-8') as f:
        if has_new_line:
            f.writelines(content)
        else:
            new_content = [i+'\n' for i in content if i is not None]
            f.writelines(new_content)
    print('save file %s successful!' % filename)


def save_jsonl_file(filename, data, indent=None):
    """
    :param filename: 输出文件名
    :param data: 数据对象，json/list
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as fp:
        for js in data:
            if indent:
                js_str = json.dumps(js, indent=indent, ensure_ascii=False)
            else:
                js_str = json.dumps(js, ensure_ascii=False)
            fp.write(js_str + '\n')
    print('save file %s successful!' % filename)


def save_json_file(filename, data, indent=2):
    """
    :param filename: 输出文件名
    :param data: 数据对象，json/list
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as fp:
        if indent:
            json.dump(data, fp, indent=indent, ensure_ascii=False)
        else:
            json.dump(data, fp, ensure_ascii=False)
    print('save file %s successful!' % filename)


def make_dir_if_needed(filename):
    abs_path = os.path.abspath(filename)
    dir_path = os.path.dirname(abs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"create dir {dir_path}")


def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"create dir {folder}")


def datetime2str():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d-%H%M%S')


# 给定文件名，和待pickle的对象，用新文件将其覆盖
def overwrite_file(filename, data):
    tmp_filename = filename + '.swp'

    if filename.endswith('jsonl'):
        func = save_jsonl_file
    elif filename.endswith('json'):
        func = save_json_file
    elif filename.endswith('txt'):
        func = save_file
    else:
        raise ValueError('undefined file format.')
    
    func(tmp_filename, data)
    
    if os.path.exists(filename):
        os.rename(filename, filename + '.old.' + datetime2str())
    os.rename(tmp_filename, filename)
    print('overwrite %s successful and backup it!' % filename)


def get_files(root_path, suffix=None):
    if not os.path.exists(root_path):
        raise FileNotFoundError(f'path {root_path} not found.')
    all_files = []
    for root, dirs, files in os.walk(root_path):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        for p in files:
            if suffix and not p.endswith(suffix):
                continue
            t = join(root, p)
            all_files.append(t)

    return all_files

def is_chinese_word(word):
    for c in word:
        if not ('\u4e00' <= c <= '\u9fa5'):
            # print(word)
            return False
    return True

def is_chinese_char(c):
    if len(c.strip()) == 1 and '\u4e00' <= c <= '\u9fa5':
        return True
    return False

def is_digit(text):
    try:
        x = int(text)
    except:
        return False
    return True