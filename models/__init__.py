import os
import json
import torch
import logging
from typing import Dict
from .grounding_model import BaseGroundingModel, UniGroundingModel
from .denoising_model import UniGroundingModel_RDropout, UniGroundingModel_MultiView, UniGroundingModel_NoisingInput
from .eta_generator import ETAGroundingGenerator
from .nn_layers import ScaledDotProductAttention, BahdanauAttention

def load_grounding_model(model_name: str, config: Dict, checkpoint: str=None) -> BaseGroundingModel:
    if model_name in ['UniGroundingModel', 'UniGrounding', 'UniG']:
        model = UniGroundingModel(config=config)
    elif model_name in ['UniGroundingModel_RDropout', 'UniGrounding_RDropout', 'UniGrounding_RDropout', 'UniG_RD']:
        model = UniGroundingModel_RDropout(config=config)
    elif model_name in ['UniGroundingModel_MultiView', 'UniGrounding_MultiView', 'UniG_MV']:
        model = UniGroundingModel_MultiView(config=config)
    elif model_name in ['UniGroundingModel_NoisingInput', 'UniGrounding_NoisingInput', 'UniG_NI']:
        model = UniGroundingModel_NoisingInput(config=config)
    else:
        raise NotImplementedError(model_name)
        
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        logging.info("Initialize model from checkpoint {} over.".format(checkpoint))
    
    return model

def load_model_from_trained(ckpt_path: str) -> BaseGroundingModel:
    config_path = os.path.join(os.path.dirname(ckpt_path), 'model_config.json')
    logging.info("Load model config from {} ...".format(config_path))    
    config: Dict = json.load(open(config_path, 'r', encoding='utf-8'))

    return load_grounding_model(model_name=config.get('model', 'UniGrounding'), config=config, checkpoint=ckpt_path)