# coding = utf-8

# chinese tokenization
import os
import jieba
from enum import Enum
from contracts.base_types import Token
from string import ascii_letters, punctuation
from utils.data_encoder import PrefixMatcher, preprocess_phrase_nl_tokens

project_path = os.path.abspath(os.path.curdir)

class CharType(int, Enum):
    UNKNOWN = -1

    LETTER = 0
    NUMBER = 1
    PUNCTUATION = 2
    CHINESE = 3
    

def get_type(c):
    if c in ascii_letters:
        return CharType.LETTER
    elif c.isdigit():
        return CharType.NUMBER
    elif c in punctuation:
        return CharType.PUNCTUATION
    elif is_chinese_word(c):
        return CharType.CHINESE
    else:
        return CharType.UNKNOWN


def is_number_string(s):
    try:
        n = float(s)
        return True
    except:
        return False


def jieba_cut(s):
    toks = [x for x in jieba.cut(s)]
    return toks


def is_chinese_word(word):
    for c in word:
        if not ('\u4e00' <= c <= '\u9fa5'):
            return False
    return True


def char_tokenization(q):
    tokens = jieba_cut(q)
    chars = []
    for w in tokens:
        w = w.strip()
        if not w: continue
        if is_chinese_word(w):
            for c in w:
                chars.append(c)
        else:
            if is_number_string(w):
                chars.append(w)
            elif len(w) > 1 and w[-1] == '%':
                chars.extend([w[:-1], w[-1]])
            else:
                small_chunk = split_pun_letter_number(w)
                chars.extend(small_chunk)
    return chars


def split_pun_letter_number(w):
    if len(w) == 1:
        return [w]
    
    chars = []
    pre = None
    text = ''
    for i, c in enumerate(w):
        if i == 0:
            pre = get_type(c)
            text += c
        else:
            cur = get_type(c)
            if cur == pre:
                text += c
            else:
                pre = cur
                chars.append(text)
                text = c
    chars.append(text)
    return chars

spm_prefix_matcher = PrefixMatcher(os.path.join(project_path, 'data', 'resources', 'spm.phrase.txt'))

def spm_tokenization(chars):
    tokens = [Token.from_json({'token': c, 'lemma': c}) for c in chars]
    tokens, idx_mappings = preprocess_phrase_nl_tokens(spm_prefix_matcher, tokens)
    tokens = [x.token for x in tokens]
    return tokens, idx_mappings
