
from copy import deepcopy
import os
from os.path import join
import shutil
from typing import Dict, List
from datetime import datetime
from easydict import EasyDict as edict

from tqdm import tqdm
import torch

from models import *
from utils import *
from inference.data_types import *
from inference.pipeline_base import *
from inference.inference_models import *
from inference.pipeline_torchscript import *
from utils.evaluate_tools import *

infer_input_names = ['input_token_ids', 'entity_indices', 'question_indices', 'column_indices', 'value_indices']


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

def convert_to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.cpu().numpy()

    if isinstance(inputs, list):
        return [convert_to_numpy(x) for x in inputs]

    if isinstance(inputs, dict):
        return { key : convert_to_numpy(val) for key, val in inputs.items() }

    raise NotImplementedError(inputs)

def convert_to_request(example: Text2SQLExample) -> NLBindingRequest:
    return NLBindingRequest(
        language=example.language,
        question_tokens=[NLToken.from_json(x.to_json()) for x in example.question.tokens],
        columns=[NLColumn.from_json(x.to_json()) for x in example.schema.columns],
        matched_values=[NLMatchedValue(column=x.column, name=x.name, tokens=[NLToken.from_json(xx.to_json()) for xx in x.tokens], start=x.start, end=x.end) for x in example.matched_values]
    )

def load_data_inputs(path: str, tokenizer: PreTrainedTokenizer, sampling_size: int = -1, infer_input_names: List[str] = infer_input_names) -> List[Dict]:
    examples: List[Text2SQLExample] = load_json_objects(Text2SQLExample, path)
    examples = [ex.ignore_unmatched_values() for ex in examples if ex.value_resolved]

    if sampling_size > 0:
        print()
        print('sampling {} examples ...'.format(sampling_size))
        examples = examples[:sampling_size] # random.sample(examples, sampling_size)

    data_iter = load_data_loader(
        examples=examples,
        tokenizer=tokenizer,
        batch_size=1,
        is_training=False,
        max_enc_length=tokenizer.model_max_length,
        n_processes=1)


    data_inputs, data_requests, cp_labels, data_examples = [], [], [], []
    for batch_inputs in data_iter:
        data_inputs += [[batch_inputs[key][0] for key in infer_input_names ]]

        data_requests += [convert_to_request(batch_inputs['example'][0])]

        data_examples += [batch_inputs['example'][0]]

        cp_label = batch_inputs['concept_labels'][0].cpu().tolist()
        cp_labels += [cp_label]

    return data_inputs, data_requests, cp_labels, data_examples



def get_binding_tag_str(binding_list):
        # Col Val O Key Unk Amb
        merge = []
        for i, js in enumerate(binding_list):
            token = js['token']
            term_type = js['term_type']

            if term_type == QuestionLabel.Null.name:
                term_type = 'O'
            term_type = term_type[:3]

            if term_type not in ['O', 'Key']:
                one = [token, term_type]
            else:
                term_type = 'O'
                one = [token, 'O']
            
            if i == 0:
                merge.append(one)
                continue
            
            if term_type == merge[-1][1]:
                merge[-1][0] += " " + token
            else:
                merge.append(one)
            
        str_list = []
        for token, term_type in merge:
            if term_type == 'O':
                text = token
            else:
                text = "[{} | {}]".format(token, term_type)
            str_list.append(text)
        
        bind_str = ' '.join(str_list)
        return bind_str


def get_binding_result_str(binding_list, tag=False):
        # Col Val O Key Unk Amb
        merge = []
        find = False

        if tag:
            str_list = []
            
            for i, js in enumerate(binding_list):
                token = js['token']
                term_type = js['term_type']

                if term_type == QuestionLabel.Null.name:
                    term_type = 'O'
                term_type = term_type[:3]

                if term_type not in ['O', 'Key']:
                    str_list += [f"[{token} | {term_type}]"]
                else:
                    str_list += [token]


            bind_str = ' '.join(str_list)
            return bind_str

        for i, js in enumerate(binding_list):
            token = js['token']
            term_type = js['term_type']

            if term_type == QuestionLabel.Null.name:
                term_type = 'O'
            term_type = term_type[:3]
            term_value = js['term_value']
            confidence = js['confidence']

            one = [token, term_type, term_value, confidence]

            if i == 0:
                merge.append(one)
                continue

            if term_type in ['Col', 'Val'] and merge[-1][1:-1] == one[1:-1]:
                merge[-1][0] += " " + token
            elif term_type in ['O', 'Unk', 'Amb'] and merge[-1][1] == one[1]:
                merge[-1][0] += " " + token
            else:
                merge.append(one)
        
        str_list = []
        for token, term_type, term_value, confidence in merge:
            if term_type in ['Col', 'Val']:
                # text = "[{} | {} | {}]".format(token, term_value, confidence)
                text = "[{} | {}]".format(token, term_value.strip())
            elif term_type in ['Unk', 'Amb']:
                text = "[{} | {}]".format(token, term_type)
            else:
                text = token
            str_list.append(text)
        
        bind_str = ' '.join(str_list)
        return bind_str


def get_concept_list(result: NLBindingResult):
    concept_list = []
    for i, x in enumerate(result.term_results):
        if x.term_score > 0.5:
            score_str = float(str(x.term_score)[:4])
            concept_list.append([x.term_type.name, x.term_value, score_str])
    return concept_list


def dump_nl_binding_predictions(requests: List[NLBindingRequest], pipeline: NLBindingInferencePipeline, saved_path: str, data_examples: List[Text2SQLExample]):
    time_costs = []
    # wramup
    result = pipeline.infer(requests[0])

    in_list = []
    out_list = []

    # saved_dir = os.path.dirname(saved_path)

    in_file = saved_path.replace('predict', 'input')
    
    i = 0
    for j, request in enumerate(tqdm(requests)):
        print('\n--------------------------------------------------------')
        example = data_examples[j]
        try:
            result = pipeline.infer(request)
        except:
            continue

        time_costs.append(result.inference_ms)
        input_js = example.to_json()

        print('UUID: {}\n'.format(example.uuid))
        print('Question: {}\n'.format(example.question.text))
        print('Dataset: {}; Schema: {}\n'.format(example.dataset, example.schema.to_string()))
        print('SQL: {}\n'.format(str(example.sql)))

        output_js = deepcopy(input_js)
        del output_js['language']

        concept_list = get_concept_list(result)
        output_js['concepts'] = concept_list


        print('Gold VS SL VS Grounding:')
        gold_binding_result = example.get_binding_result_json_list()
        output_js['gold_binding_result'] = gold_binding_result
        result.print_binding_result(gold_binding_result)

        
        sl_binding_js_list = result.export_sequence_labeling_json()
        output_js['sl_binding_result'] = sl_binding_js_list

        grounding_binding_js_list = result.export_binding_json()
        output_js['sl_grounding_binding_result'] = grounding_binding_js_list

        in_list.append(input_js)
        out_list.append(output_js)
        i += 1
    
    save_json_file(in_file, in_list)
    save_json_file(saved_path, out_list)
    return out_list


def print_bad_case(data, examples):
    gbinds = [x.gold_binding_result for x in data]
    pbinds = [x.sl_grounding_binding_result for x in data]

    p_sl_binds = [x.sl_binding_result for x in data]

    correct_all = 0
    total_all = 0

    print('\n------------------------\n')
    print('bad case:')
    
    for idx, (g, p) in enumerate(zip(gbinds, pbinds)):
        correct = get_sentence_correct(g, p)
        correct_all += int(correct)
        total_all += 1

        if not correct:
            example = examples[idx]
            print('UUID: {}'.format(example.uuid))
            print('Question: {}'.format(example.question.text))
            print('Dataset: {}; Schema: {}'.format(example.dataset, example.schema.to_string()))
            print('SQL: {}'.format(str(example.sql)))

            print()
            gold_bind_str = get_binding_result_str(g)
            print(f"Gold: {gold_bind_str}")

            pred_bind_str = get_binding_result_str(p)
            print(f"Pred: {pred_bind_str}")

            sl_binds = p_sl_binds[idx]
            pred_bind_tag_str = get_binding_tag_str(sl_binds)
            print(f"Tag : {pred_bind_tag_str}")

            print('\n------------------------\n')
    
    col_acc = (correct_all * 100) / total_all
    
    acc_str = f"sentence accuracy: {str(col_acc)[:4]}%   {correct_all}/{total_all}"
    print(acc_str)


def print_evaluate_performance(data):
    gbinds = [x.gold_binding_result for x in data]
    pbinds = [x.sl_grounding_binding_result for x in data]
    p_sl_binds = [x.sl_binding_result for x in data]


    gconcepts = [get_gold_concepts(x.gold_binding_result) for x in data]
    pconcepts = [x.concepts if 'concepts' in x else [] for x in data]


    col_acc, acc_str = get_sentence_accuracy(gbinds, pbinds)
    print(acc_str)

    col_acc, acc_str = get_tag_accuracy(gbinds, pbinds, tag='Column')
    print(acc_str)

    col_acc, acc_str = get_tag_accuracy(gbinds, pbinds, tag='Value')
    print(acc_str)

    col_acc, acc_str = get_tag_accuracy(gbinds, p_sl_binds, tag='Unknown')
    print(acc_str)

    col_acc, acc_str = get_concept_accuracy(gconcepts, pconcepts, tags=['Column'])
    print(acc_str)

    col_acc, acc_str = get_concept_accuracy(gconcepts, pconcepts, tags=['Value'])
    print(acc_str)



def infer_torchscript(args):

    model_dir = os.path.dirname(args.checkpoint)
    # Load model
    model = GroundingInferenceModel.from_trained(args.checkpoint)

    # Load data inputs
    if args.sampling:
        data_inputs, data_requests, _, data_examples = load_data_inputs(args.data_path, tokenizer=model.tokenizer, sampling_size=20)
    else:
        data_inputs, data_requests, _, data_examples = load_data_inputs(args.data_path, tokenizer=model.tokenizer)
    
    dummy_input = data_inputs[0]
    print('sample data_inputs[0]  shape:')
    for input_name, input_value in zip(infer_input_names, dummy_input):
        print(input_name, " = ", input_value.size())
    print()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    basename = os.path.basename(args.data_path)

    predict_save_path = join(out_dir, basename + '.predict.json')

    pipeline = NLBindingTorchScriptPipeline(model_dir=model_dir, model=model, \
        greedy_linking=True, threshold=0.2, use_gpu=torch.cuda.is_available() and args.gpu)

    out_list = dump_nl_binding_predictions(data_requests, pipeline, predict_save_path, data_examples)

    data = [edict(i) for i in out_list]

    print_evaluate_performance(data)

    print_bad_case(data, data_examples)

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint', type=str, required=True)
    parser.add_argument('-do_val', '--do_validation', action='store_true')
    parser.add_argument('-data_path', '--data_path', default='data/wikisql_label/dev.preproc.json')
    parser.add_argument('-out_dir', '--out_dir', default='./output')
    parser.add_argument('-sampling', '--sampling', action='store_true')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    args = parser.parse_args()

    infer_torchscript(args)