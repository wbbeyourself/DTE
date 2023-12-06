from multiprocessing import pool
import os
import json
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from contracts.text2sql import QuestionLabel
from sklearn.metrics.pairwise import cosine_similarity
from models import BaseGroundingModel, ScaledDotProductAttention

class GroundingInferenceModel(BaseGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(pretrain_model_name=config['pretrained_model'], torch_script=True)
        self.config = config
        self.hidden_size = config['hidden_size']

        self.num_labels = config['num_labels']

        self.linear_softmax_classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.encoder.config.hidden_size, self.num_labels),
        )

        self.crf = CRF(self.num_labels)

        self.linear_enc_out = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout'])
        )

        self.cls_to_keywords = nn.Sequential(
            nn.Linear(self.hidden_size, config['num_keywords'] * self.hidden_size),
            nn.Tanh()
        )

        self.concept_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.grounding_scorer = ScaledDotProductAttention(self.hidden_size)

    def get_max_similar_target(self, q_vecs, _v_vecs, _c_vecs, seq_labels):
        col_tag_value = QuestionLabel.Column.value
        val_tag_value = QuestionLabel.Value.value

        def get_tgt_spans(labels, tgt_value):
            spans = []
            found = False
            for i, tag in enumerate(labels):
                if tag != tgt_value:
                    found = False
                else:
                    if found:
                        spans[-1].append(i)
                    else:
                        found = True
                        spans.append([i])
            return spans
        
        def get_pooled_vecs(vecs, spans):
            pool_vecs = []
            for idxs in spans:
                v = vecs[idxs[0]:(idxs[-1]+1)]
                v = torch.mean(v, dim=0)
                pool_vecs.append(v.cpu().tolist())
            return pool_vecs
        
        def get_max_score_and_idx(sim_score):
            # print('sim_score:')
            # print(sim_score)
            tgt_ids = np.argmax(sim_score, axis=-1)
            # print(f"tgt_ids: {tgt_ids}")
            tgt_scores = np.max(sim_score, axis=-1)
            # print(f"tgt_scores: {tgt_scores}")
            id_score_tuple = [(int(_id), _score) for _id, _score in zip(tgt_ids, tgt_scores)]
            return id_score_tuple
        
        c_spans = get_tgt_spans(seq_labels, col_tag_value)
        v_spans = get_tgt_spans(seq_labels, val_tag_value)

        
        tgt_entities = [(-1, -1.0)] * len(seq_labels)
        
        c_vecs = _c_vecs.cpu().tolist() if _c_vecs is not None else None
        v_vecs = _v_vecs.cpu().tolist() if _v_vecs is not None else None
        
        qc_vecs = []
        if c_spans and c_vecs is not None:
            qc_vecs = get_pooled_vecs(q_vecs, c_spans)
            sim_score = cosine_similarity(qc_vecs, c_vecs)
            col_id_score_tuple = get_max_score_and_idx(sim_score)
            for span, (_id, _score) in zip(c_spans, col_id_score_tuple):
                for k in span:
                    tgt_entities[k] = (int(_id), _score)

        qv_vecs = []
        if v_spans and v_vecs is not None:
            qv_vecs = get_pooled_vecs(q_vecs, v_spans)
            sim_score = cosine_similarity(qv_vecs, v_vecs)
            val_id_score_tuple = get_max_score_and_idx(sim_score)
            for span, (_id, _score) in zip(v_spans, val_id_score_tuple):
                for k in span:
                    tgt_entities[k] = (int(_id), _score)

        # print(f"tgt_entities : {tgt_entities}")
        return tgt_entities

    def get_pool_vec_by_span(self, vecs, indices):
        # squeeze batch
        seq_tensors = vecs.squeeze(0)
        fetched_outputs = []
        for s, e in indices:
            # print(f"s,e : {s},{e}")
            hidden = seq_tensors[s+1:e+1]
            # print(f"hidden shape； {hidden.shape}")
            hidden = torch.mean(hidden, dim=0)
            # print(f"hidden shape； {hidden.shape}")
            fetched_outputs.append(hidden)
        pooled_vec = torch.stack(fetched_outputs, dim=0)
        return pooled_vec
    
    
    def get_concept_mask(self, cp_scores, num_columns, num_values, value2column_index):
        """
        Using concept_scores to generate mask for later calculation.
        """
        column_mask = [0] * num_columns
        value_mask = None
        if num_values > 0:
            value_mask = [0] * num_values
        
        num_keywords = len(cp_scores) - num_columns - num_values

        threshold = 0.3

        for idx, term_score in enumerate(cp_scores):
            term_score = cp_scores[idx]
            if term_score < threshold:
                continue

            term_type, term_index, term_value = None, None, None
            if idx < num_keywords:
                pass
            elif idx < num_columns + num_keywords:
                term_type = 'Column'
                term_index = idx - num_keywords
                column_mask[term_index] = 1
            else:
                if num_values > 0:
                    term_type = 'Value'
                    term_index = idx - num_columns - num_keywords
                    # get the column index of this value
                    col_idx = value2column_index[term_index]
                    column_mask[col_idx] = 1
                    value_mask[term_index] = 1
        
        # print(f"column_mask : {column_mask}")

        column_mask = torch.tensor(column_mask)
        if num_values > 0:
            value_mask = torch.tensor(value_mask)
     
        return column_mask, value_mask
    
    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor, column_indices: torch.LongTensor=None, value_indices: torch.LongTensor=None, value2column_index: Dict = {}) -> Tuple[torch.Tensor, torch.Tensor]:
        input_token_ids = input_token_ids.unsqueeze(0)
        attention_mask = input_token_ids.ne(self.tokenizer.pad_token_id)

        encoder_outputs = self.encoder(
            input_token_ids,
            attention_mask=attention_mask
        )[0]

        raw_entity_outputs = encoder_outputs[:, entity_indices]
        entity_outputs = self.linear_enc_out(raw_entity_outputs)

        keyword_outputs = self.cls_to_keywords(self.linear_enc_out(encoder_outputs[:, 0])).view(encoder_outputs.size(0), -1, self.hidden_size)
        concept_outputs = torch.cat([keyword_outputs, entity_outputs], dim=1)

        concept_scores = self.concept_scorer.forward(concept_outputs).squeeze(2).squeeze(0)
        
        raw_question_outputs = encoder_outputs[:, question_indices]
        question_outputs = self.linear_enc_out(raw_question_outputs)

        grounding_scores = self.grounding_scorer.forward(query=concept_outputs, key=question_outputs, mask=None).squeeze(0)

        grounding_scores = grounding_scores[:, :-1]

        # print(f"question_indices : {question_indices.shape}")
        # print(f"grounding_scores : {grounding_scores.shape}")

        # crf label
        pure_question_indices = question_indices[:-1]

        # print(f"pure_question_indices shape: {pure_question_indices.shape}")
        pure_question_outputs = encoder_outputs[:, pure_question_indices]

        question_logits = self.linear_softmax_classifier(pure_question_outputs)

        # print(f"question_logits shape : {question_logits.shape}")
        seq_first_question_logits = question_logits.transpose(0, 1)
        
        seq_labels = self.crf.decode(seq_first_question_logits)
        # type(seq_labels) : <class 'list'>
        # print(f"type(seq_labels) : {type(seq_labels)}")

        seq_labels = seq_labels[0]

        q_vecs = pure_question_outputs.squeeze(0)

        num_columns, num_values = 0, 0
        
        is_c_valid = False
        if column_indices is not None:
            is_c_valid = True
            num_columns = len(column_indices)
        
        is_v_valid = False
        if value_indices is not None:
            is_v_valid = True
            num_values = len(value_indices)
        
        c_vecs = None
        if is_c_valid:
            c_vecs = self.get_pool_vec_by_span(encoder_outputs, column_indices)
        
        v_vecs = None
        if is_v_valid:
            v_vecs = self.get_pool_vec_by_span(encoder_outputs, value_indices)

        # before calculation , mask vector using concept_scores
        column_mask, value_mask = self.get_concept_mask(concept_scores, num_columns, num_values, value2column_index)
        column_mask = column_mask.unsqueeze(1)
        column_mask = column_mask.to(c_vecs.device)
        c_vecs = column_mask * c_vecs
        
        if is_v_valid:
            value_mask = value_mask.unsqueeze(1)
            value_mask = value_mask.to(v_vecs.device)
            v_vecs = value_mask * v_vecs
        
        tgt_entities = self.get_max_similar_target(q_vecs, v_vecs, c_vecs, seq_labels)
        
        tgt_entities = torch.tensor(tgt_entities)

        # print(f"seq_labels len : {len(seq_labels)}")
        # print(f"seq_labels : {seq_labels}")
        seq_labels = torch.tensor(seq_labels)

        return concept_scores, grounding_scores, seq_labels, tgt_entities

    @classmethod
    def from_trained(cls, ckpt_path: str):
        config_path = os.path.join(os.path.dirname(ckpt_path), 'model_config.json')
        print("Load model config from {} ...".format(config_path))
        config = json.load(open(config_path, 'r', encoding='utf-8'))

        model = cls(config)
        print('load model from checkpoint {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        model.eval()
        return model

class GroundingInferenceModelForORT(GroundingInferenceModel):
    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, question_indices: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_token_ids = input_token_ids.unsqueeze(0)
        attention_mask = input_token_ids.ne(self.tokenizer.pad_token_id).to(torch.long)

        encoder_outputs = self.encoder.forward(
            input_token_ids,
            attention_mask=attention_mask,
        )[0]

        entity_outputs = self.linear_enc_out(encoder_outputs[:, entity_indices])
        keyword_outputs = self.cls_to_keywords(self.linear_enc_out(encoder_outputs[:, 0])).view(encoder_outputs.size(0), -1, self.hidden_size)
        concept_outputs = torch.cat([keyword_outputs, entity_outputs], dim=1)

        concept_scores = self.concept_scorer.forward(concept_outputs).squeeze(2).squeeze(0)
        question_outputs = self.linear_enc_out(encoder_outputs[:, question_indices])

        grounding_scores = self.grounding_scorer.forward(query=concept_outputs, key=question_outputs, mask=None).squeeze(0)

        return concept_scores, grounding_scores[:, :-1]