# %%
from utils import *
from trainer import Trainer
import os
# Force the process to see only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"


# %%

# Model paths

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
FINETUNED_MODEL_PATH = "../finetuned-qwen-coder-3b-attnonly-lr02" + "/final_model"  # Update this path if needed
model_name = "Qwen/Qwen2.5-3B-Instruct"

base_hooked, finetuned_hooked = get_models(BASE_MODEL_NAME, FINETUNED_MODEL_PATH, model_name)

# %%
all_tokens = load_and_merge_two_datasets()

torch.manual_seed(42)

# Generate a random permutation of indices
shuffled_indices = torch.randperm(all_tokens.size(0))

# Shuffle the tensor along the 0-axis
all_tokens = all_tokens[shuffled_indices]

# %%
default_cfg = {
    "seed": 42,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 50_000_000,
    #"num_tokens": 4097,
    "l1_coeff": 5,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_hooked.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.18.hook_resid_pre",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_hooked, finetuned_hooked, all_tokens)
trainer.train()
# %%