import random
from typing import List

from .grounding_model import *

class UniGroundingModel_NoisingInput(UniGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(config)
    
    def forward(self, input_token_ids: torch.LongTensor, entity_indices: torch.LongTensor, **kwargs):
        if kwargs.get('is_training', False):
            input_token_ids = self.random_erasing_inputs(input_token_ids, p_erasing=0.2, **kwargs)

        return super().forward(input_token_ids, entity_indices, **kwargs)
    
    def random_erasing_inputs(self, input_token_ids: torch.LongTensor, p_erasing: float, **kwargs):
        for b_idx in range(len(input_token_ids)):
            for c_idx, (start, _, end) in enumerate(kwargs['blocks'][b_idx]):
                if random.random() < p_erasing:
                    input_token_ids[b_idx, start:end+1] = self.tokenizer.unk_token_id
        return input_token_ids

class UniGroundingModel_RDropout(UniGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(config)
    
    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = super().compute_loss(inputs, outputs)

        cp_scores2 = self.forward(
            input_token_ids=inputs['input_token_ids'],
            entity_indices=inputs['entity_indices'],
            concept_predict_only=True
        )['concept_scores']

        cp_loss2 = F.binary_cross_entropy_with_logits(
            input=torch.logit(cp_scores2, eps=1e-4),
            target=inputs['concept_labels'].to(torch.float) * (1.0 - self.label_smoothing),
            weight=inputs['concept_mask'].to(torch.float)
        )

        denosing_loss = F.mse_loss(
            input=outputs['concept_scores'],
            target=cp_scores2
        )

        outputs['denosing_loss'] = denosing_loss
        outputs['cp_loss'] = (outputs['cp_loss'] + cp_loss2) / 2
        
        outputs['loss'] = outputs['cp_loss'] + denosing_loss * 0.2 + outputs.get('grounding_loss', 0.0) * inputs.get('grounding_loss_weight', 1.0)
        return outputs

class UniGroundingModel_MultiView(UniGroundingModel):
    def __init__(self, config: Dict):
        super().__init__(config)
    
    def _generate_single_view_attention_mask(self, input_token_ids: torch.Tensor, blocks: List[List[Tuple]]):
        """
        Generate attention mask, which columns can't see each other
        """
        assert input_token_ids.ndim == 2
        attention_mask: torch.BoolTensor = input_token_ids.ne(self.tokenizer.pad_token_id)\
            .unsqueeze(1).repeat_interleave(input_token_ids.size(1), dim=1)
        
        for b_idx in range(len(input_token_ids)):
            col_spans = blocks[b_idx]
            for i in range(len(col_spans)):
                for j in range(i + 1, len(col_spans)):
                    (start1, _, end1), (start2, _, end2) = col_spans[i], col_spans[j]
                    # print("Batch Index = {}, Column Index: {}: {}".format(b_idx, i, " ".join(self.tokenizer.convert_ids_to_tokens(input_token_ids[b_idx, start:end+1].tolist()))))
                    attention_mask[b_idx, start1:end1+1,start2:end2+1] = 0
                    attention_mask[b_idx, start2:end2+1,start1:end1+1] = 0

        return attention_mask
    
    def compute_loss(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = super().compute_loss(inputs, outputs)
        if not inputs['is_training']:
            return outputs

        single_view_attention_mask = self._generate_single_view_attention_mask(
            input_token_ids=inputs['input_token_ids'],
            blocks=inputs['blocks']
        )

        cp_scores2 = self.forward(
            input_token_ids=inputs['input_token_ids'],
            entity_indices=inputs['entity_indices'],
            attention_mask=single_view_attention_mask,
            concept_predict_only=True
        )['concept_scores']

        cp_loss2 = F.binary_cross_entropy_with_logits(
            input=torch.logit(cp_scores2, eps=1e-4),
            target=inputs['concept_labels'].to(torch.float) * (1.0 - self.label_smoothing),
            weight=inputs['concept_mask'].to(torch.float)
        )

        denosing_loss = F.mse_loss(
            input=outputs['concept_scores'],
            target=cp_scores2
        )

        outputs['denosing_loss'] = denosing_loss
        outputs['cp_loss'] = (outputs['cp_loss'] + cp_loss2) / 2
        
        outputs['loss'] = outputs['cp_loss'] + denosing_loss + outputs.get('grounding_loss', 0.0) * inputs.get('grounding_loss_weight', 1.0)
        return outputs