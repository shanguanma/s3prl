import tempfile
from pathlib import Path
from typing import List

import torch

import s3prl
#from s3prl.upstream.hubert.hubert_model import (
#    HubertPretrainingConfig,
#)
#from s3prl.upstream.imls_ssl.imls_ssl_model import (
#    ImlsHubertConfig,
#    ImlsHubertModel,
#)
from .hubert_with_llama_model import LLaMAHubertConfig, LLaMAHubertModel,  HubertPretrainingConfig 
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.util.download import _urls_to_filepaths

def freeze_model_layer(model, freeze_hubert_layer_nums:int, freeze_llama: bool = True):
    freeze_keys=[]
    for k, params in model.named_parameters():
        if f'feature_extractor.' in k:
            freeze_keys.append(k)
        elif f'post_extract_proj.' in k:
            freeze_keys.append(k)
        elif f'encoder.pos_conv.' in k:
            freeze_keys.append(k)
        elif freeze_llama and f'llama.' in k:
            freeze_keys.append(k)

        elif freeze_hubert_layer_nums>0:
            for n in range(freeze_hubert_layer_nums):
                if f'encoder.layers.{n}.' in k:
                    freeze_keys.append(k)

    print(f"freeze_keys: {freeze_keys}!!!!")
    for name, params in model.named_parameters():
        if name in freeze_keys:
            params.requires_grad = False


def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str = None):
    from fairseq.data.dictionary import Dictionary

    state, cfg = load_fairseq_ckpt(fairseq_source)

    dicts: List[Dictionary] = state["task_state"]["dictionaries"]
    symbols = [dictionary.symbols for dictionary in dicts]

    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
        "dictionaries_symbols": symbols,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(output_state, output_path)

## for expert.py
def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in [
        "task_cfg",
        "model_cfg",
        "model_weight",
        "dictionaries_symbols",
    ]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )
    print(f"task_cfg: {ckpt_state['task_cfg']}")
    print(f"model_cfg: {ckpt_state['model_cfg']}")
    ## why I will add the below command, we need modfiy model file  and remove BaseFairseqModel register_model 
    task_cfg = merge_with_parent(HubertPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(LLaMAHubertConfig, ckpt_state["model_cfg"])
    model = LLaMAHubertModel(model_cfg, task_cfg, ckpt_state["dictionaries_symbols"])
    
    model.remove_pretraining_modules()

    #freeze_model_layer(model, ckpt_state["model_cfg"]["freeze_hubert_layer_nums"], freeze_llama=True)

    print(f"llama layer weight grad in model: {ckpt_state['model_weight']['llama.layers.0.attn.c_attn.weight'].requires_grad} !!!!")
    model.load_state_dict(ckpt_state["model_weight"])
    
    return model, task_cfg


def load_converted_model_without_llama(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in [
        "task_cfg",
        "model_cfg",
        "model_weight",
        "dictionaries_symbols",
    ]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )
    print(f"task_cfg: {ckpt_state['task_cfg']}")
    print(f"model_cfg: {ckpt_state['model_cfg']}")
    ## why I will add the below command, we need modfiy model file  and remove BaseFairseqModel register_model
    task_cfg = merge_with_parent(HubertPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(LLaMAHubertConfig, ckpt_state["model_cfg"])
    model = LLaMAHubertModel(model_cfg, task_cfg, ckpt_state["dictionaries_symbols"])

    #model.remove_pretraining_modules()
    model.remove_pretraining_modules_include_llama()
    ## remove related llama layer
    #del ckpt_state["model_weight"]["llama"]
    # llama.layers.0.rms_1.scale
    # llama.layers.0.attn.c_attn.weight
    # llama.layers.0.attn.c_proj.weight
    # llama.layers.0.rms_2.scale
    # llama.layers.0.mlp.c_fc1.weight
    # llama.layers.0.mlp.c_fc2.weight
    # llama.layers.0.mlp.c_proj.weight
    # llama.norm.scale
    # llama_dim_mapper1.weight
    #llama_dim_mapper2.weight
    del ckpt_state["model_weight"]['llama.layers.0.rms_1.scale']
    del ckpt_state["model_weight"]['llama.layers.0.attn.c_attn.weight']
    del ckpt_state["model_weight"]['llama.layers.0.attn.c_proj.weight']
    del ckpt_state["model_weight"]['llama.layers.0.rms_2.scale']
    del ckpt_state["model_weight"]['llama.layers.0.mlp.c_fc1.weight']
    del ckpt_state["model_weight"]['llama.layers.0.mlp.c_fc2.weight']
    del ckpt_state["model_weight"]['llama.layers.0.mlp.c_proj.weight']
    del ckpt_state["model_weight"]['llama.norm.scale']
    del ckpt_state["model_weight"]["llama_dim_mapper1.weight"]
    del ckpt_state["model_weight"]["llama_dim_mapper2.weight"]
    ## remove unused label embedding
    del ckpt_state["model_weight"]["label_embs_concat"]
    #freeze_model_layer(model, ckpt_state["model_cfg"]["freeze_hubert_layer_nums"], freeze_llama=True)
    print(f"remove llama layer !!!")
    #print(f"llama layer weight grad in model: {ckpt_state['model_weight']['llama.layers.0.attn.c_attn.weight'].requires_grad} !!!!")
    model.load_state_dict(ckpt_state["model_weight"])

    return model, task_cfg
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fairseq_ckpt")
    parser.add_argument(
        "--output_dir", default=Path(s3prl.__file__).parent.parent / "converted_ckpts"
    )
    args = parser.parse_args()

    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    load_and_convert_fairseq_ckpt(
        args.fairseq_ckpt, Path(args.output_dir) / f"{Path(args.fairseq_ckpt).stem}.pt"
    )
