"""
Model prediction implementation based on TorchScript
"""
import os
import torch
from inference.pipeline_base import NLBindingInferencePipeline
from inference.inference_models import GroundingInferenceModel

class NLBindingTorchScriptPipeline(NLBindingInferencePipeline):
    def __init__(self,
        model_dir: str,
        model: GroundingInferenceModel,
        greedy_linking: bool,
        threshold: float=0.2,
        num_threads: int=8,
        use_gpu: bool = torch.cuda.is_available()
    ) -> None:
        super().__init__(model_dir, greedy_linking=greedy_linking, threshold=threshold)

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        self.model = model.to(self.device)
        self.model.eval()

    def run_model(self, **inputs):
        input_token_ids = torch.as_tensor(inputs['input_token_ids'], dtype=torch.long, device=self.device)
        entity_indices = torch.as_tensor(inputs['entity_indices'], dtype=torch.long, device=self.device)
        column_indices = torch.as_tensor(inputs['column_indices'], dtype=torch.long, device=self.device)
        value_indices = torch.as_tensor(inputs['value_indices'], dtype=torch.long, device=self.device)
        question_indices = torch.as_tensor(inputs['question_indices'], dtype=torch.long, device=self.device)
        value2column_index = inputs['value2column_index']

        if min(value_indices.shape) == 0:
            value_indices = None
        if min(column_indices.shape) == 0:
            column_indices = None

        cp_scores, grounding_scores, seq_labels, tgt_entities = self.model(input_token_ids, entity_indices, question_indices, column_indices, value_indices, value2column_index)

        
        cp_scores = cp_scores.cpu().tolist()
        grounding_scores = grounding_scores.cpu().tolist()

        if 'spm_idx_mappings' in inputs:
            idx_mappings = inputs['spm_idx_mappings']
            raw_grounding_scores = [[-100 for _ in idx_mappings] for _ in grounding_scores]
            for k, _ in enumerate(grounding_scores):
                for idx1, idx2 in enumerate(idx_mappings):
                    if idx1 > 0 and idx_mappings[idx1 - 1] == idx_mappings[idx1]:
                        continue
                    raw_grounding_scores[k][idx1] = grounding_scores[k][idx2]
            grounding_scores = raw_grounding_scores

        
        # print(f"seq_labels : {seq_labels}")
        # print(f"tgt_entities : {tgt_entities}")
        return {
            'cp_scores': cp_scores,
            'grounding_scores': grounding_scores,
            'seq_labels': seq_labels,
            'tgt_entities': tgt_entities
        }
