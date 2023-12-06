import os
import onnxruntime as ort
from .pipeline_base import NLBindingInferencePipeline

class NLBindingONNXRuntimePipeline(NLBindingInferencePipeline):
    def __init__(self, model_dir: str, greedy_linking: bool, threshold: float=0.2, num_threads: int=8, use_gpu: bool = False) -> None:
        super().__init__(model_dir, greedy_linking=greedy_linking, threshold=threshold)

        model_file = "nl_binding.onnx"
        model_path = os.path.join(model_dir, model_file)
        sess_options = ort.SessionOptions()
        self.sess = ort.InferenceSession(model_path, sess_options)
        print("Load ONNX model from {} over.".format(model_path))

    def run_model(self, **inputs):
        ort_inputs = {
            'input_token_ids': inputs['input_token_ids'],
            'entity_indices': inputs['entity_indices'],
            'question_indices': inputs['question_indices']
        }
        ort_outputs = self.sess.run(None, ort_inputs)
        cp_scores, grounding_scores = ort_outputs[0].tolist(), ort_outputs[1].tolist()

        if 'spm_idx_mappings' in inputs:
            idx_mappings = inputs['spm_idx_mappings']
            raw_grounding_scores = [[-100 for _ in idx_mappings] for _ in grounding_scores]
            for k, _ in enumerate(grounding_scores):
                for idx1, idx2 in enumerate(idx_mappings):
                    if idx1 > 0 and idx_mappings[idx1 - 1] == idx_mappings[idx1]:
                        continue
                    raw_grounding_scores[k][idx1] = grounding_scores[k][idx2]
            grounding_scores = raw_grounding_scores

        return {
            'cp_scores': cp_scores,
            'grounding_scores': grounding_scores
        }