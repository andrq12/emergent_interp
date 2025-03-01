from utils import *

from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download

from typing import NamedTuple

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("checkpoints")

class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor 
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        
        # Number of shared features (S)
        self.n_shared = cfg.get("n_shared_features", d_hidden // 16)
        # Number of standard features (F)
        self.n_standard = d_hidden - self.n_shared
        
        # Ratio of shared to standard feature sparsity penalty
        self.lambda_ratio = cfg.get("lambda_ratio", 0.15)  # default to middle of 0.1-0.2 range
        
        # Create encoder weights
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        )
        
        # Create decoder weights - shared features use the same weights for both models
        # For shared features (indices 0:n_shared), we need only one set of weights that will be broadcast
        self.W_dec_shared = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.n_shared, 1, d_in, dtype=self.dtype
                )
            )
        )
        
        # For standard features (indices n_shared:), we keep separate weights per model
        self.W_dec_standard = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.n_standard, 2, d_in, dtype=self.dtype
                )
            )
        )
        
        # Make norm of W_dec 0.1 for each column
        self.W_dec_shared.data = (
            self.W_dec_shared.data / self.W_dec_shared.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        self.W_dec_standard.data = (
            self.W_dec_standard.data / self.W_dec_standard.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        
        # Initialize encoder weights based on decoder weights
        # First initialize with random values
        self.W_enc.data = torch.nn.init.normal_(self.W_enc.data) * 0.01
        
        # Then set the shared portion of the encoder to be the transpose of the shared decoder
        shared_dec_expanded = self.W_dec_shared.data.expand(-1, 2, -1)  # Expand to both models
        shared_enc = einops.rearrange(
            shared_dec_expanded,
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.W_enc.data[:, :, :self.n_shared] = shared_enc[:, :, :self.n_shared]
        
        # Set the standard portion of the encoder to be the transpose of the standard decoder
        standard_enc = einops.rearrange(
            self.W_dec_standard.data,
            "d_hidden_std n_models d_model -> n_models d_model d_hidden_std",
        )
        self.W_enc.data[:, :, self.n_shared:] = standard_enc
        
        # Bias terms
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((2, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        # Split activations into shared and standard parts
        acts_shared = acts[:, :self.n_shared]
        acts_standard = acts[:, self.n_shared:]
        
        # Process shared features - broadcast the same weights to both models
        shared_dec = einops.einsum(
            acts_shared,
            self.W_dec_shared,
            "batch d_hidden_shared, d_hidden_shared n_models d_model -> batch n_models d_model",
        )
        
        # Process standard features - use separate weights for each model
        standard_dec = einops.einsum(
            acts_standard,
            self.W_dec_standard,
            "batch d_hidden_std, d_hidden_std n_models d_model -> batch n_models d_model",
        )
        
        # Combine the results
        return shared_dec + standard_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance

        per_token_l2_loss_A = (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A

        per_token_l2_loss_B = (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B

        # Split activations for separate L1 penalties
        acts_shared = acts[:, :self.n_shared]
        acts_standard = acts[:, self.n_shared:]
        
        # For shared features, we use the same weights for both models, so we calculate the norm once
        shared_decoder_norms = self.W_dec_shared.squeeze(1).norm(dim=-1)  # [d_hidden_shared]
        
        # For standard features, sum across models as in original code
        standard_decoder_norms = self.W_dec_standard.norm(dim=-1)  # [d_hidden_std, n_models]
        total_standard_decoder_norm = einops.reduce(standard_decoder_norms, 'd_hidden_std n_models -> d_hidden_std', 'sum')
        
        # Calculate L1 losses with different lambda values
        # Calculate the weighted activations for each feature set
        weighted_acts_shared = (acts_shared * shared_decoder_norms[None, :])
        weighted_acts_standard = (acts_standard * total_standard_decoder_norm[None, :])

        # Reduce along the hidden dimension for each batch example
        l1_per_example_shared = weighted_acts_shared.sum(-1) * self.lambda_ratio
        l1_per_example_standard = weighted_acts_standard.sum(-1)

        # Sum these contributions (now they're both [batch] shaped)
        l1_per_example = l1_per_example_shared + l1_per_example_standard

        # Take the mean across the batch
        l1_loss = l1_per_example.mean(0)
        
        # L0 loss remains the same (count of non-zero activations)
        l0_loss = (acts > 0).float().sum(-1).mean()

        return LossOutput(
            l2_loss=l2_loss, 
            l1_loss=l1_loss,  # Add this for compatibility
            l0_loss=l0_loss, 
            explained_variance=explained_variance, 
            explained_variance_A=explained_variance_A, 
            explained_variance_B=explained_variance_B
        )
        
    def get_total_loss(self, x):
        """Helper method to compute the total loss for optimization"""
        losses = self.get_losses(x)
        # Combine L1 losses - standard features use the full lambda penalty from config
        lambda_penalty = self.cfg.get("lambda", 1.0)
        total_loss = losses.l2_loss + lambda_penalty * (losses.l1_loss_standard + losses.l1_loss_shared)
        return total_loss

    def create_save_dir(self):
        base_dir = Path("checkpoints")
        version_list = [
            int(file.name.split("_")[1])
            for file in list(SAVE_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)
            
        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance

    @classmethod
    def load(cls, version_dir, checkpoint_version):
        save_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints") / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self