from typing import List, Dict
from contracts.text2sql import BindingItem

from inference.data_types import *
from inference.greedy_linker import greedy_link

from contracts import Text2SQLExample, QuestionLabel, All_Agg_Op_Keywords

confidence_threshold = 0.2
_All_Agg_Op_Keywords = ["Max", "Min", "Sum", "Avg", "Count", "!=", ">", ">=", "<", "<=" ]
_Data_Type_Mappings = [NLDataType.String, NLDataType.DateTime, NLDataType.Integer, NLDataType.Double, NLDataType.Boolean]

def load_request(obj: Dict) -> NLBindingRequest:
    question_tokens = [NLToken(x['token'], lemma=x['lemma']) for x in obj['question']['tokens']]
    columns = []
    for col_obj in obj['schema']['columns']:
        column = NLColumn(
            name=col_obj['name'],
            tokens=[NLToken(token=x['token'], lemma=x['lemma']) for x in col_obj['tokens']],
            data_type=_Data_Type_Mappings[col_obj['data_type']]
        )

        if len(column.tokens) == 0:
            continue
        columns += [column]

    matched_values = []
    for val_obj in obj['matched_values']:
        value = NLMatchedValue(
            name=val_obj['name'],
            tokens=[NLToken(x['token'], x['lemma']) for x in val_obj['tokens']],
            column=val_obj['column'],
            start=val_obj['span']['start'],
            end=val_obj['span']['end'])

        matched_values += [value]

    return NLBindingRequest(
        question_tokens=question_tokens,
        columns=columns,
        matched_values=matched_values,
        language=LanguageCode.from_json(obj['language'])
    )

def get_term_results(model_outputs, request: NLBindingRequest):
        cp_scores, grounding_scores = model_outputs['cp_scores'], model_outputs['grounding_scores']
        if len(cp_scores) != len(grounding_scores) or len(request.question_tokens) != len(grounding_scores[0]):
            raise ValueError('NLBinding Output Error: Invalid Model Predictions')

        term_results: List[NLBindingTermResult] = []
        num_columns, num_values = len(request.columns), len(request.matched_values)
        num_keywords = len(cp_scores) - num_columns - num_values

        for idx, term_score in enumerate(cp_scores):
            term_score = cp_scores[idx]
            if term_score < confidence_threshold:
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

def get_question_labels(result: List[NLBindingToken]):
    question_labels = []
    for item in result:
        label_value = QuestionLabel.get_value(item.term_type.name)
        question_labels.append(label_value)
    return question_labels


def get_hand_craft_labels(binding_result: List[BindingItem]):
    hand_craft_labels = []
    for i, item in enumerate(binding_result):
        lab = None
        if item.term_type == QuestionLabel.Unknown.name:
            lab = QuestionLabel.Unknown.value
        elif item.term_type == QuestionLabel.Ambiguity.name:
            lab = QuestionLabel.Ambiguity.value
        hand_craft_labels.append(lab)
    return hand_craft_labels


def scores_to_labels(batch_cp_scores, batch_grounding_scores, batch_examples):
    batch_size = batch_cp_scores.size(0)
    batch_labels = []
    max_seq_len = -1
    for idx in range(batch_size):
        cp_scores = batch_cp_scores[idx]
        grounding_scores = batch_grounding_scores[idx]
        example: Text2SQLExample = batch_examples[idx]

        try:
            js = example.to_json()
            request = load_request(js)

            concept_len = len(All_Agg_Op_Keywords) + len(request.columns) + len(request.matched_values)
            seq_len = len(request.question_tokens)

            model_outputs = {}
            model_outputs['cp_scores'] = cp_scores[:concept_len]

            raw_grounding_scores = [[grounding_scores[i][j] for j in range(seq_len)] for i in range(concept_len)]
            model_outputs['grounding_scores'] = raw_grounding_scores

            term_results = get_term_results(model_outputs, request)

            binding_tokens = greedy_link(request=request, term_results=term_results, threshold=confidence_threshold)
            question_labels = get_question_labels(binding_tokens)

            if example.question.tokens[-1].lemma in ['?', '.', '？', '。']:
                question_labels[-1] = QuestionLabel.Null.value
        except Exception as e:
            print(f"example.uuid : {example.uuid}")
            print(f"example.question.text : {example.question.text}")
            raise ValueError(e)
        
        hand_craft_labels = get_hand_craft_labels(example.binding_result)

        # merge labels
        merged_labels = []
        assert len(question_labels) == len(hand_craft_labels)
        for export, hand in zip(question_labels, hand_craft_labels):
            if hand:
                merged_labels.append(hand)
            else:
                merged_labels.append(export)

        if len(merged_labels) > max_seq_len:
            max_seq_len = len(merged_labels)

        # print(f'question: {example.question.text}')
        # print(f'merged_labels: {merged_labels}')
        batch_labels.append(merged_labels)
    
    # padding
    for labels in batch_labels:
        pad_len = max_seq_len - len(labels)
        if pad_len > 0:
            labels += [0] * pad_len
    
    return batch_labels