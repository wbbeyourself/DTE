"""
NLBinding pipeline: tokenization, model prediction, post-prcessing.
"""
import abc
from copy import deepcopy
import os
import torch
import json
import traceback
from typing import List, Dict
from datetime import datetime
from transformers import AutoTokenizer

from contracts.text2sql import QuestionLabel

from inference.data_types import *
from inference.data_encoder import NLDataEncoder
from inference.greedy_linker import greedy_link

_All_Agg_Op_Keywords = ["Max", "Min", "Sum", "Avg", "Count", "!=", ">", ">=", "<", "<=" ]

class NLBindingInferencePipeline:
    def __init__(self, model_dir:str, greedy_linking: bool, threshold: float=0.2) -> None:
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.data_encoder = NLDataEncoder(self.tokenizer, spm_phrase_path=os.path.join(model_dir, "spm.phrase.txt"))
        self.threshold = threshold
        self.greedy_linking = greedy_linking

    @abc.abstractmethod
    def run_model(self, **inputs) -> Dict[str, List]:
        raise NotImplementedError()

    def predict(self, data: str) -> str:
        start = datetime.now()

        try:
            try:
                request_obj = json.loads(data)
            except: # pylint: disable=bare-except
                raise NLModelError( # pylint: disable=raise-missing-from
                    error_code=StatusCode.invalid_input,
                    message="Input String Json Decoder Failed: {}".format(traceback.format_exc())
                )

            try:
                request = NLBindingRequest.from_json(request_obj)
            except:
                raise NLModelError( # pylint: disable=raise-missing-from
                    error_code=StatusCode.invalid_input,
                    message="Request Json Deserialization Failed: {}".format(traceback.format_exc())
                )

            result = self.infer(request)

        except NLModelError as e:
            result = NLBindingResult(
                status_code=e.error_code,
                message=e.message,
                inference_ms=(datetime.now() - start).microseconds // 1000,
                term_results=[],
                binding_tokens=None
            )

        except: # pylint: disable=bare-except
            result = NLBindingResult(
                status_code=StatusCode.internal_error,
                message="Unknown Internal Error: {}".format(traceback.format_exc()),
                inference_ms=(datetime.now() - start).microseconds // 1000,
                term_results=[],
                binding_tokens=None
            )

        return json.dumps(result.to_json())

    def infer(self, request: NLBindingRequest) -> NLBindingResult:
        start = datetime.now()

        model_inputs = self.encode_request(request)
        model_outputs = self.run_model(**model_inputs)

        _seq_labels = model_outputs['seq_labels']
        _tgt_entities = model_outputs['tgt_entities']
        input_tokens = model_inputs['input_tokens']

        # print(f"input_tokens : {input_tokens}")

        seq_labels = [QuestionLabel.get_name_by_value(v) for v in _seq_labels]

        if request.question_tokens[-1].lemma in ['.', '?']:
            seq_labels[-1] = QuestionLabel.get_name_by_value(0)

        tgt_entities = []
        for i, (_id, _score) in enumerate(_tgt_entities):
            if isinstance(_id, torch.Tensor):
                _id = _id.int()
            if _id == -1:
                tgt_entities.append(TagEntity.from_json(["", -1.0]))
            else:
                # print(f"_id: {_id}")
                # print(f"_id type: {type(_id)}")
                tag_value = _seq_labels[i]
                if tag_value == QuestionLabel.Column.value:
                    if _id < len(request.columns):
                        col = request.columns[_id].name
                        score = float('{:.3f}'.format(_score))
                        tgt_entities.append(TagEntity.from_json([col, score]))
                    else:
                        print("error: index out of range")
                        tgt_entities.append(TagEntity.from_json(["", -1.0]))
                elif tag_value == QuestionLabel.Value.value:
                    if _id < len(request.matched_values):
                        val = str(request.matched_values[_id])
                        score = float('{:.3f}'.format(_score))
                        tgt_entities.append(TagEntity.from_json([val, score]))
                    else:
                        print("error: index out of range")
                        tgt_entities.append(TagEntity.from_json(["", -1.0]))

        term_results = self.get_term_results(model_outputs, request)

        binding_tokens = None
        if self.greedy_linking:
            binding_tokens: List[NLBindingToken] = greedy_link(request=request, term_results=term_results, threshold=self.threshold)

        inference_ms = (datetime.now() - start).microseconds // 1000
        
        # relocate seq_labels and tgt_entities
        # label need to convert too
        if 'spm_idx_mappings' in model_inputs:
            idx_mappings = model_inputs['spm_idx_mappings']
            raw_seq_labels = ['' for _ in idx_mappings]
            raw_tgt_entities = [None for _ in idx_mappings]

            raw_binding_tokens = [None for _ in idx_mappings]

            for idx1, idx2 in enumerate(idx_mappings):
                raw_seq_labels[idx1] = seq_labels[idx2]
                raw_tgt_entities[idx1] = tgt_entities[idx2]

                q_tokens: List[NLToken] = request.question_tokens
                bind_token: NLBindingToken = deepcopy(binding_tokens[idx2])
                bind_token.token = q_tokens[idx1].token
                bind_token.lemma = q_tokens[idx1].token

                if bind_token.term_type == NLBindingType.Null:
                    bind_token.term_value = bind_token.token
                elif bind_token.term_type == NLBindingType.Column:
                    pass
                elif bind_token.term_type == NLBindingType.Value:
                    pass
                elif bind_token.term_type == NLBindingType.Keyword:
                    pass
                raw_binding_tokens[idx1] = bind_token
            
            seq_labels = raw_seq_labels
            tgt_entities = raw_tgt_entities
            binding_tokens = raw_binding_tokens
        
        return NLBindingResult(
            status_code=StatusCode.succeed,
            message=None,
            inference_ms=inference_ms,
            term_results=term_results,
            binding_tokens=binding_tokens,
            tgt_entities=tgt_entities,
            seq_labels=seq_labels
        )

    def encode_request(self, request: NLBindingRequest):
        input_tokens, input_token_ids, spm_idx_mappings, indices, value2column_index = \
            self.data_encoder.encode(request.question_tokens, request.columns, request.matched_values, request.language)
        

        if len(input_token_ids) > self.tokenizer.model_max_length:
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Input Token Length Out of {}".format(self.tokenizer.model_max_length),
                )

        if len(indices['question']) == 0 :
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Empty Question"
                )

        if len(indices['column']) == 0:
            raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: Empty Columns"
                )

        for span in indices['value']:
            if span is None:
                raise NLModelError(
                    error_code=StatusCode.invalid_input,
                    message="NLBinding Input Error: None Value Span"
                )

        column_span_indices = [[start, end] for start, end in indices['column']]
        value_span_indices = [[start, end] for start, end in indices['value']]

        column_indices = [start for start, _ in indices['column']]
        value_indices = [start for start, _ in indices['value']]

        question_indices = [start for start, _ in indices['question']]

        return {
            'input_token_ids': input_token_ids,
            'input_tokens': input_tokens,
            'entity_indices': column_indices + value_indices,
            'column_indices': column_span_indices,
            'value_indices': value_span_indices,
            'question_indices': question_indices,
            'spm_idx_mappings': spm_idx_mappings,
            'value2column_index': value2column_index
        }

    def get_term_results(self, model_outputs, request: NLBindingRequest):
        cp_scores, grounding_scores = model_outputs['cp_scores'], model_outputs['grounding_scores']
        if len(cp_scores) != len(grounding_scores) or len(request.question_tokens) != len(grounding_scores[0]):
            raise NLModelError(
                    error_code= StatusCode.invalid_output,
                    message="NLBinding Output Error: Invalid Model Predictions"
                )

        term_results: List[NLBindingTermResult] = []
        num_columns, num_values = len(request.columns), len(request.matched_values)
        num_keywords = len(cp_scores) - num_columns - num_values

        for idx, term_score in enumerate(cp_scores):
            term_score = cp_scores[idx]
            if term_score < self.threshold:
                continue

            term_type, term_index, term_value = None, None, None
            if idx < num_keywords:
                term_type = NLBindingType.Keyword
                term_index = idx
                term_value = _All_Agg_Op_Keywords[idx]
            elif idx < num_columns + num_keywords:
                term_type = NLBindingType.Column
                term_index = idx - num_keywords
                term_value = request.columns[term_index].name
            else:
                term_type = NLBindingType.Value
                term_index = idx - num_columns - num_keywords
                term_value = str(request.matched_values[term_index])

            term_result = NLBindingTermResult(
                term_type=term_type,
                term_index=term_index,
                term_value=term_value,
                term_score=term_score,
                grounding_scores=grounding_scores[idx]
            )

            term_results += [term_result]

        return term_results
