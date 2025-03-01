# %%
import os
from IPython import get_ipython

ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Import stuff
import einops
import json
import argparse

from datasets import load_dataset
from pathlib import Path
import plotly.express as px
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import torch
import numpy as np
from transformer_lens import HookedTransformer
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from functools import partial

from IPython.display import HTML

from transformer_lens.utils import to_numpy
import pandas as pd

from html import escape
import colorsys


import wandb

import plotly.graph_objects as go

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis",
     "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid",
     "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth"
}

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    fig = px.imshow(to_numpy(tensor), color_continuous_midpoint=0.0,labels={"x":xaxis, "y":yaxis}, **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    fig.show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(y=to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, return_fig=False, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    if return_fig:
        return fig
    fig.show(renderer)

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def bar(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.bar(
        y=to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis},
        template="simple_white",
        **kwargs).show(renderer)

def create_html(strings, values, saturation=0.5, allow_different_length=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", "<br/>").replace("\t", "&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))

# crosscoder stuff

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg    


import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

def load_and_merge_two_datasets(sample_size=10_000):
    """
    Uses HF streaming with .shuffle(...) + .take(...) to get a random sample 
    without iterating over the entire dataset. Then merges and tokenizes.
    """
    # Local paths for caching
    cache_pt_path = "../data/monology-lmsys-merged.pt"
    cache_hf_path = "../data/monology-lmsys-merged.hf"

    # Model tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

    try:
        print("Loading existing tokenized data from disk...")
        all_tokens = torch.load(cache_pt_path)
        print(f"Found cached file at {cache_pt_path}")
    except FileNotFoundError:
        print("No cached file found. Streaming and sampling data from HF...")

        # -----------------------------------------------------
        # 1) SAMPLE monology/pile-uncopyrighted
        # -----------------------------------------------------
        print(f"\nSampling {sample_size} from monology/pile-uncopyrighted...")
        monology_stream = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True
        )
        # Shuffle with a buffer (approx. random) and take the first `sample_size` rows
        monology_shuffled = monology_stream.shuffle(seed=42, buffer_size=100_000)
        monology_subset = monology_shuffled.take(sample_size)

        # Materialize
        monology_samples = list(monology_subset)
        monology_dataset = Dataset.from_list(monology_samples)

        # -----------------------------------------------------
        # 2) SAMPLE lmsys/lmsys-chat-1m
        # -----------------------------------------------------
        print(f"\nSampling {sample_size} from lmsys/lmsys-chat-1m...")
        lmsys_stream = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True
        )
        lmsys_shuffled = lmsys_stream.shuffle(seed=42, buffer_size=100_000)
        lmsys_subset = lmsys_shuffled.take(sample_size)

        # Materialize
        lmsys_samples = list(lmsys_subset)
        lmsys_dataset = Dataset.from_list(lmsys_samples)

        # -----------------------------------------------------
        # 3) Convert chat fields to "text"
        #    Adjust based on actual columns. Some versions have "messages", 
        #    some have "conversation", etc.
        # -----------------------------------------------------
        def merge_chat_messages(example):
            """
            Combine role + content into a single 'text' field.
            For instance, if the dataset has:
              example["messages"] = [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
              ]
            """
            start_token = "<|im_start|>"
            end_token   = "<|im_end|>"
            if "messages" in example and example["messages"] is not None:
                text_parts = []
                for turn in example["messages"]:
                    role = turn["role"]
                    content = turn["content"]
                    text_parts.append(f"{start_token}{role}\n{content}{end_token}")
                example["text"] = "\n".join(text_parts)
            elif "conversation" in example and example["conversation"] is not None:
                text_parts = []
                for turn in example["conversation"]:
                    role = turn["role"]
                    content = turn["content"]
                    text_parts.append(f"{start_token}{role}\n{content}{end_token}")
                example["text"] = "\n".join(text_parts)
            return example

        print("Converting chat fields to 'text' in LMSYS dataset...")
        lmsys_dataset = lmsys_dataset.map(merge_chat_messages, desc="Formatting chat text")

        # -----------------------------------------------------
        # 4) Combine
        # -----------------------------------------------------
        combined_dataset = concatenate_datasets([monology_dataset, lmsys_dataset])

        # -----------------------------------------------------
        # 5) Tokenize
        # -----------------------------------------------------
        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=1024  # or your desired context size
            )

        print("Tokenizing combined dataset...")
        combined_dataset = combined_dataset.map(tokenize_fn, batched=True, desc="Tokenizing")

        # -----------------------------------------------------
        # 6) Save
        # -----------------------------------------------------
        combined_dataset.save_to_disk(cache_hf_path)
        combined_dataset.set_format(type="torch", columns=["input_ids"])
        all_tokens = combined_dataset["input_ids"]

        torch.save(all_tokens, cache_pt_path)
        print(f"Saved merged tokenized dataset to {cache_pt_path}")

    return all_tokens

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel


def get_models(BASE_MODEL_NAME, FINETUNED_MODEL_PATH, model_name, sae=False):
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
    config.attention_implementation = "eager" # or "flash_attention_2" if available
    config.sliding_window = None # Disable sliding window attention

    # Load base model
    print(f"Loading base model from {BASE_MODEL_NAME}...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, add_eos_token=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True,
        #attn_implementation="eager",
        device_map="cuda:0",
        config=config,
        #quantization_config=bnb_config,
        #load_in_4bit=True
    ).to("cuda:0")

    # Load finetuned model
    print(f"Loading finetuned model from {FINETUNED_MODEL_PATH}...")

    base_for_finetune_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True,
        #attn_implementation="eager",
        device_map="cuda:0",
        config=config,
        #quantization_config=bnb_config,
        #load_in_4bit=True
    ).to("cuda:1")

    finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, add_eos_token=True)

    peft_model = PeftModel.from_pretrained(base_for_finetune_model, FINETUNED_MODEL_PATH)

    # Merge the LoRA weights into the base model and unload the adapter
    finetuned_model = peft_model.merge_and_unload()
    peft_model.to("cpu")
    base_for_finetune_model.to("cpu")    


    del base_for_finetune_model
    del peft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    base_model.eval()

    finetuned_model.eval()


    def convert_hf_to_hooked(name, tokenizer, model, device, sae=False):
        """Convert a HuggingFace model to HookedTransformer format."""
        print(f"Converting {name} to HookedTransformer...")
        
        # Create a HookedTransformer instance directly from the HF model

        if not sae:

            return HookedTransformer.from_pretrained(
                model_name=model_name,
                hf_model=model.to(device),
                tokenizer=tokenizer,
                device=device,
                fold_ln=False,  # Keep layer norms separate for analysis
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=False
            )

        else:

            return HookedSAETransformer.from_pretrained(
                model_name=model_name,
                hf_model=model.to(device),
                tokenizer=tokenizer,
                device=device,
                fold_ln=False,  # Keep layer norms separate for analysis
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=False
            )


    base_hooked = convert_hf_to_hooked("base_qwen", base_tokenizer, base_model, device=torch.device("cuda:0"), sae=sae).to("cuda:0")

    finetuned_hooked = convert_hf_to_hooked("finetune_qwen", finetuned_tokenizer, finetuned_model, device=torch.device("cuda:0"), sae=sae).to("cuda:0")

    return base_hooked, finetuned_hooked

from sae_lens import SAE, HookedSAETransformer, SAEConfig
