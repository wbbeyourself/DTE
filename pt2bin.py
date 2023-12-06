
import os
from typing import Dict, List

import torch

from models import *
from utils import *
from inference import *
from inference.inference_models import GroundingInferenceModel
from utils.inference_data_loader import get_input_list

infer_input_names = ['input_token_ids', 'entity_indices', 'question_indices', 'column_indices', 'value_indices']


def convert_to_request(example: Text2SQLExample) -> NLBindingRequest:
    return NLBindingRequest(
        language=example.language,
        question_tokens=[NLToken.from_json(x.to_json()) for x in example.question.tokens],
        columns=[NLColumn.from_json(x.to_json()) for x in example.schema.columns],
        matched_values=[NLMatchedValue(column=x.column, name=x.name, tokens=[NLToken.from_json(xx.to_json()) for xx in x.tokens], start=x.start, end=x.end) for x in example.matched_values]
    )

def load_data_inputs(path: str, tokenizer: PreTrainedTokenizer, sampling_size: int = 500, infer_input_names: List[str] = infer_input_names) -> List[Dict]:
    examples: List[Text2SQLExample] = load_json_objects(Text2SQLExample, path)
    examples = [ex.ignore_unmatched_values() for ex in examples if ex.value_resolved]

    if sampling_size > 0:
        print('random sampling {} examples ...'.format(sampling_size))
        examples = examples[:sampling_size] # random.sample(examples, sampling_size)

    data_iter = load_data_loader(
        examples=examples,
        tokenizer=tokenizer,
        batch_size=1,
        is_training=False,
        max_enc_length=tokenizer.model_max_length,
        n_processes=1)


    data_inputs, data_requests, cp_labels = [], [], []
    for batch_inputs in data_iter:
        data_inputs += [[batch_inputs[key][0] for key in infer_input_names ]]

        data_requests += [convert_to_request(batch_inputs['example'][0])]

        cp_label = batch_inputs['concept_labels'][0].cpu().tolist()
        cp_labels += [cp_label]

    return data_inputs, data_requests, cp_labels


def do_pt_to_bin(pt_file, out_dir):
    model = GroundingInferenceModel.from_trained(pt_file)
    sample_file = './cpu_bin/sample.preproc.json'
    # data_inputs, data_requests, cp_labels = load_data_inputs(sample_file, tokenizer=model.tokenizer, sampling_size=1)
    model_dir = os.path.dirname(pt_file)

    data_inputs = get_input_list(sample_file, model_dir, model)

    sample_input = data_inputs[0]

    # print(f"model.device: {model.device}")

    # CPU model
    script_cpu_model = torch.jit.trace(model, sample_input)
    script_cpu_path = os.path.join(out_dir, 'nl_binding.script.bin')
    torch.jit.save(script_cpu_model, script_cpu_path)
    print("Dump scripted cpu model into {}".format(script_cpu_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', help='checkpoint', type=str, required=True)
    args = parser.parse_args()

    pt_file = args.checkpoint
    out_dir = './cpu_bin'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    do_pt_to_bin(pt_file, out_dir)