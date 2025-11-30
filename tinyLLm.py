import torch
from enum import Enum
from typing import Union,List, Dict,Optional,Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class EngineType(Enum):
    HF="hf"
    VLLM="vllm"

class TinyLLM:
    def __init__(
        self,
        engine_type: EngineType,
        engine_params: Dict[str, Any],
        tokenizer_params: Dict[str, Any],
        sampling_params: Dict[str,Any]
    ):
        self.engine_type=engine_type
        self.sampling_params=sampling_params
        # Listen to tokenizer_params first
        t_model= tokenizer_params.get("pretrained_model_name_or_path", engine_params.get("pretrained_model_name_or_path"))
        # For simplicity, we use the model_name from engine_params if tokenizer_name isn't specific.
        if not t_model:
            # Fallback if specific key isn't standard
            t_model=list(engine_params.values())[0]

        print(f"[{self.engine_type.value.upper()}] Initializing...")

        if self.engine_type==EngineType.HF:
            self._init_hf(engine_params,tokenizer_params)
        elif self.engine_type==EngineType.VLLM:
            self._init_vllm(engine_params,tokenizer_params)
        else:
            raise ValueError("Invalid engine type")
    
    def _init_hf(self,engine_params:Dict,tokenizer_params:Dict):
        # 1. Load Tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(**tokenizer_params)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

        # 2. Load Model
        if "device_map" not in engine_params:
            engine_params["device_map"]="auto"
        self.model=AutoModelForCausalLM.from_pretrained(**engine_params)
        self.model.eval()

    def _init_vllm(self,engine_params:Dict,tokenizer_params:Dict):
        # 1. Load Tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(**tokenizer_params)
        self.model=LLM(**engine_params)

    def generate(self,inputs:Union[str,List[str],Dict, List[Dict], List[List[Dict]]])->Union[str,List[str]]:
        is_batch=False
        is_chat=False
        final_prompts=[]

        if isinstance(inputs,str):
            final_prompts=[inputs]
            is_batch=False
            is_chat=False
        elif isinstance(inputs,list) and len(inputs)>0 and isinstance(inputs[0],str):
            final_prompts=inputs
            is_batch=True
            is_chat=False
        
        elif isinstance(inputs, dict) or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict)):
            convo=[inputs] if isinstance(inputs,dict) else inputs
            prompt_str=self.tokenizer.apply_chat_template(convo, tokenize=False,add_generation_prompt=True)
            final_prompts=[prompt_str]
            is_batch=False
            is_chat=True
            