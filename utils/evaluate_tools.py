from easydict import EasyDict as edict
from contracts.schemas import *
from contracts.text2sql import *

def get_gold_concepts(binds):
    concepts = []
    d = {}
    for i, js in enumerate(binds):
        js = edict(js)
        if js.term_type != 'Null':
            term_type = js.term_type
            term_value = js.term_value
            confidence = js.confidence
            if term_value not in d:
                one = [term_type, term_value, confidence]
                concepts.append(one)
                d[term_value] = 1
    return concepts


# def get_sentence_correct(gold_bind_list, pred_bind_list, TAGS=['Column', 'Value', 'Null']):
def get_sentence_correct(gold_bind_list, pred_bind_list):
    total = 0
    correct = 0
    assert len(gold_bind_list) == len(pred_bind_list), "len_gold != len_pred"
    g_bind_objs: List[BindingItem] = [BindingItem.from_json(x) for x in gold_bind_list]
    p_bind_objs: List[BindingItem] = [BindingItem.from_json(x) for x in pred_bind_list]
    for g, p in zip(g_bind_objs, p_bind_objs):
        assert g.token == p.token
        total += 1
        # ignore term_value
        if p.term_type in ['O', 'Null', 'Keyword', 'Key', 'Coreterm', 'Cor'] and g.term_type in ['O', 'Null', 'Keyword', 'Key', 'Coreterm', 'Cor']:
            correct += 1
            continue
        pv = p.term_value.strip()
        gv = g.term_value.strip()
        if p.term_type == g.term_type and pv == gv:
            correct += 1
    is_same = correct == total
    return is_same


def get_sentence_accuracy(g_binds, p_binds):
    correct_all = 0
    total_all = 0
    
    for g, p in zip(g_binds, p_binds):
        correct = get_sentence_correct(g, p)
        correct_all += int(correct)
        total_all += 1
    
    col_acc = (correct_all * 100) / total_all
    
    acc_str = f"sentence accuracy: {str(col_acc)[:4]}%   {correct_all}/{total_all}"
    return col_acc, acc_str


def get_concept_correct(gold_concept_list, pred_concept_list, tags=['Column']):
    total = 0
    correct = 0
    
    golds = [x[1] for x in gold_concept_list if x[0] in tags]
    preds = [x[1] for x in pred_concept_list if x[0] in tags]
    
    gold_set = set(golds)
    pred_set = set(preds)
    
    common = gold_set & pred_set
    missing = list(gold_set - common)
    
    correct = len(common)
    total = len(gold_set)
    
    return correct, total, missing


def get_concept_accuracy(g_concepts, p_concepts, tags=['Column']):
    correct_all = 0
    total_all = 0
    
    for g, p in zip(g_concepts, p_concepts):
        correct, total, missing = get_concept_correct(g, p, tags)
        correct_all += correct
        total_all += total
    
    col_acc = (correct_all * 100) / total_all
    
    acc_str = f"{tags} concept accuracy: {str(col_acc)[:4]}%   {correct_all}/{total_all}"
    return col_acc, acc_str


def get_tag_correct(gold_bind_list, pred_bind_list, tag='Column'):
    total = 0
    correct = 0
    is_value_considered = True
    if tag in ['Unknown', 'Ambiguity', 'Coreterm']:
        is_value_considered = False
    assert len(gold_bind_list) == len(pred_bind_list), "len_gold != len_pred"
    g_bind_objs: List[BindingItem] = [BindingItem.from_json(x) for x in gold_bind_list]
    p_bind_objs: List[BindingItem] = [BindingItem.from_json(x) for x in pred_bind_list]
    for g, p in zip(g_bind_objs, p_bind_objs):
        assert g.token == p.token
        if g.term_type == tag:
            total += 1
            if p.term_type == tag:
                if is_value_considered:
                    if p.term_value == g.term_value:
                        correct += 1
                else:
                    correct += 1
    return correct, total


def get_tag_accuracy(g_binds, p_binds, tag='Column'):
    # all term_type should in ['Null', 'Keyword', 'Table', 'Column', 'Value', 'Ambiguity', 'Unknown']
    TAG_SET = ['Null', 'Keyword', 'Table', 'Column', 'Value', 'Ambiguity', 'Unknown']
    assert tag in TAG_SET, "invalid tag " + tag
    
    correct_all = 0
    total_all = 0
    
    for g, p in zip(g_binds, p_binds):
        correct, total = get_tag_correct(g, p, tag)
        correct_all += correct
        total_all += total
    
    
    if total_all > 0:
        col_acc = (correct_all * 100) / total_all
        acc_str = f"{tag} accuracy: {str(col_acc)[:4]}%   {correct_all}/{total_all}"
    else:
        col_acc = 0
        acc_str = f"no {tag} tags found !!!"
    return col_acc, acc_str