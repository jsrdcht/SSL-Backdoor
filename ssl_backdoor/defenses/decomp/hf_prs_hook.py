
import torch

class PRSLogger(object):
    def __init__(self, model, device):
        self.device = device
        self.attentions = [] # Will store [B, H, D]
        self.mlps = []       # Will store [B, D]
        self.post_ln_mean = None
        self.post_ln_std = None
        self.model = model
        self.hooks = []
        
        # Determine model dim and heads from config
        self.embed_dim = model.vision_model.config.hidden_size
        self.num_heads = model.vision_model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

    @torch.no_grad()
    def compute_attentions(self, module, input, output):
        # input[0] is [Batch, Seq_Len, Embed_Dim]
        # We only care about CLS token (index 0)
        x = input[0][:, 0, :] # [B, Embed_Dim]
        
        # Decompose into heads
        # x is [B, H * D_head]
        x_reshaped = x.reshape(-1, self.num_heads, self.head_dim) # [B, H, D_head]
        
        # Weight is [Out_Features, In_Features] = [Embed_Dim, Embed_Dim]
        # Reshape to [Embed_Dim, H, D_head]
        weight = module.weight # [Embed_Dim, Embed_Dim]
        weight_reshaped = weight.reshape(self.embed_dim, self.num_heads, self.head_dim)
        
        # Einsum: b h d, m h d -> b h m
        # Calculate contribution of each head to the output vector
        # output_h = x_h @ weight_h.T
        output_h = torch.einsum('b h d, m h d -> b h m', x_reshaped, weight_reshaped) # [B, H, Embed_Dim]
        
        # Add bias distributed across heads
        if module.bias is not None:
            bias = module.bias # [Embed_Dim]
            output_h += bias.view(1, 1, -1) / self.num_heads
            
        self.attentions.append(output_h.cpu())

    @torch.no_grad()
    def compute_mlps(self, module, input, output):
        # output is [Batch, Seq_Len, Embed_Dim]
        # Take CLS token
        cls_out = output[:, 0, :] # [B, Embed_Dim]
        self.mlps.append(cls_out.cpu())

    @torch.no_grad()
    def compute_pre_ln_output(self, module, input, output):
        # Output of pre-layernorm (R_0): initial residual stream state
        x = output
        if x.dim() == 3:  # [B, Seq, D] -> take CLS
            x = x[:, 0, :]
        elif x.dim() != 2:  # expected pooled [B, D]
            raise ValueError(f"Unexpected pre-layernorm output shape: {tuple(x.shape)}")
        self.mlps.append(x.cpu())

    @torch.no_grad()
    def compute_post_ln_stats(self, module, input, output):
        # This hook captures the state before the final LayerNorm
        # We ONLY use this to compute the mean/std for PRS normalization
        # We do NOT add this to mlps, as it is the sum of all components.
        x = input[0]
        if x.dim() == 3:  # [B, Seq, D] -> take CLS
            x = x[:, 0, :]
        elif x.dim() != 2:  # expected pooled [B, D]
            raise ValueError(f"Unexpected post_layernorm input shape: {tuple(x.shape)}")

        # self.mlps.append(x.cpu()) # REMOVED: This was causing double counting
        
        # Also compute mean and std for normalization
        # LayerNorm(x) = (x - mean) / std * gamma + beta
        # We need mean and std of x
        self.post_ln_mean = x.mean(dim=-1, keepdim=True).cpu() # [B, 1]
        self.post_ln_std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + module.eps).cpu() # [B, 1]

    def _normalize_mlps(self):
        # self.mlps is [B, L+1, D]
        # Normalization logic from original prs_hook.py
        
        # Convert list to tensor
        mlps_tensor = torch.stack(self.mlps, dim=1).to(self.device) # [B, L+1, D]
        attentions_tensor = torch.stack(self.attentions, dim=1).to(self.device) # [B, L, H, D]
        
        len_intermediates = attentions_tensor.shape[1] + mlps_tensor.shape[1]
        
        post_ln_mean = self.post_ln_mean.to(self.device).unsqueeze(1) # [B, 1, 1]
        post_ln_std = self.post_ln_std.to(self.device).unsqueeze(1)   # [B, 1, 1]
        
        # Mean centering
        # Original: self.mlps - self.post_ln_mean / len_intermediates
        # Note: Original code divides mean by len_intermediates. 
        # Ideally, Decomposition implies: Sum(Components) = Input_to_LN
        # So Mean(Sum(Components)) = Sum(Mean(Components)) = Input_Mean
        # So each component's "share" of the mean is Input_Mean / N.
        
        mean_centered = mlps_tensor - post_ln_mean / len_intermediates
        
        # Apply LN weight
        # model.visual.post_layernorm.weight
        ln_weight = self.model.vision_model.post_layernorm.weight.detach().to(self.device)
        ln_bias = self.model.vision_model.post_layernorm.bias.detach().to(self.device)
        
        weighted_mean_centered = ln_weight * mean_centered
        
        # Divide by std
        weighted_mean_by_std = weighted_mean_centered / post_ln_std
        
        # Add bias share
        bias_term = ln_bias / len_intermediates
        
        post_ln = weighted_mean_by_std + bias_term
        
        return post_ln # This is the normalized component in the residual stream

    def _normalize_attentions(self):
        attentions_tensor = torch.stack(self.attentions, dim=1).to(self.device) # [B, L, H, D]
        mlps_tensor = torch.stack(self.mlps, dim=1).to(self.device)
        
        len_intermediates = attentions_tensor.shape[1] + mlps_tensor.shape[1]
        
        # In original code, for spatial attention, normalization_term = n * h
        # Here we already summed over spatial dimensions (implicitly via out_proj inputs).
        # We effectively have [B, L, H, D] where D is Embed_Dim.
        # But wait, original code has [B, L, N, H, D] and normalization term N*H.
        # And sums over N later.
        
        # My attentions are [B, L, H, D].
        # Each entry is the contribution of one Head.
        # The sum over Heads gives the Layer Output.
        # So the "share" of the mean for each Head should be (Mean / (Layers + MLPs)) / Heads ?
        
        # Original code (non-spatial):
        # normalization_term = h
        # mean_centered = self.attentions - post_ln_mean / (len_intermediates * normalization_term)
        
        normalization_term = self.num_heads
        
        post_ln_mean = self.post_ln_mean.to(self.device).unsqueeze(1).unsqueeze(2) # [B, 1, 1, 1]
        post_ln_std = self.post_ln_std.to(self.device).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, 1]
        
        mean_centered = attentions_tensor - post_ln_mean / (len_intermediates * normalization_term)
        
        ln_weight = self.model.vision_model.post_layernorm.weight.detach().to(self.device)
        ln_bias = self.model.vision_model.post_layernorm.bias.detach().to(self.device)
        
        weighted_mean_centered = ln_weight * mean_centered
        weighted_mean_by_std = weighted_mean_centered / post_ln_std
        
        bias_term = ln_bias / (len_intermediates * normalization_term)
        
        post_ln = weighted_mean_by_std + bias_term
        
        return post_ln

    @torch.no_grad()
    def finalize(self, representation):
        # representation is [B, Output_Dim] (Projected)
        # We need to project our components as well.
        # HF CLIP Visual Projection: self.model.visual_projection
        
        # Check if projection exists
        if hasattr(self.model, 'visual_projection') and self.model.visual_projection is not None:
            proj = self.model.visual_projection.weight.detach().to(self.device).T # [Embed_Dim, Output_Dim]
        elif representation.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Representation dimension ({representation.shape[-1]}) does not match "
                f"embedding dimension ({self.embed_dim}) and `visual_projection` is not found or None. "
                "Cannot project components to match representation."
            )
        else:
            proj = torch.eye(self.embed_dim).to(self.device)

        projected_attentions = self._normalize_attentions() @ proj
        projected_mlps = self._normalize_mlps() @ proj
        
        # Normalize by representation norm
        norm = representation.norm(dim=-1, keepdim=True).detach()
        
        # projected_attentions: [B, L, H, Output_Dim]
        # projected_mlps: [B, L+1, Output_Dim]
        
        return (
            projected_attentions / norm.unsqueeze(1).unsqueeze(2),
            projected_mlps / norm.unsqueeze(1)
        )

    def reinit(self):
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def hook_prs_logger(model, device):
    """Hooks a projected residual stream logger to the HF CLIP model."""
    prs = PRSLogger(model, device)
    
    # Hook pre-layernorm output (R_0)
    pre_ln = getattr(model.vision_model, "pre_layrnorm", None) or getattr(
        model.vision_model, "pre_layernorm", None
    )
    if pre_ln is None:
        raise AttributeError(
            "Cannot find vision pre-layernorm module: expected `vision_model.pre_layrnorm` "
            "(HuggingFace CLIP) or `vision_model.pre_layernorm`."
        )
    handle = pre_ln.register_forward_hook(prs.compute_pre_ln_output)
    prs.hooks.append(handle)

    # Hook Attention Output Projections
    for i, layer in enumerate(model.vision_model.encoder.layers):
        # Hook out_proj
        handle = layer.self_attn.out_proj.register_forward_hook(prs.compute_attentions)
        prs.hooks.append(handle)
        
        # Hook MLP second FC layer (output of MLP)
        handle = layer.mlp.fc2.register_forward_hook(prs.compute_mlps)
        prs.hooks.append(handle)
        
    # Hook post-layernorm input for LN stats (mean/std) only
    handle = model.vision_model.post_layernorm.register_forward_hook(prs.compute_post_ln_stats)
    prs.hooks.append(handle)
    
    return prs
