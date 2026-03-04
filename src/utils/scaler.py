import torch

class AFMScaler:
    def __init__(self, exp_cfg):
        """
        Args:
            exp_cfg: Experiment configuration object
        """

        if exp_cfg.data.min_max_scaling:
            self.method = "min_max"
        else:
            raise ValueError("Unsupported scaling method or scaling not enabled.")

    def scale(self, x: torch.Tensor):
        """
        Raw -> Model Input (0~1 Normalized)
        Returns:
            scaled_x: Normalized tensor
            min_val: Tensor of min values (B, C, 1, 1) or scalar
            max_val: Tensor of max values (B, C, 1, 1) or scalar
        """
        if self.method != "min_max":
            return x, None, None

        # Save original shape for later reshaping
        original_shape = x.shape
        if x.dim() == 2:   # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
        
        # --- Vectorized Min/Max Calculation ---
        # flatten: (B, C, H, W) -> (B, C, H*W)
        flattened = x.flatten(-2) 
        
        # min/max values: (B, C) -> (B, C, 1, 1) for broadcasting
        min_val = flattened.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        max_val = flattened.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)

        # Broadcasting 
        numerator = x - min_val
        denominator = max_val - min_val + 1e-9
        scaled_x = numerator / denominator

        return scaled_x.view(original_shape), min_val, max_val

    def descale(self, x: torch.Tensor, min_val, max_val) -> torch.Tensor:
        """
        Model Output -> Raw (Physical Value)
        Args:
            x: Scaled tensor
            min_val: Original min values from scale()
            max_val: Original max values from scale()
        """
        if self.method != "min_max":
            return x
            
        if min_val is None or max_val is None:
            return x

        descaled_x = x * (max_val - min_val) + min_val
        
        return descaled_x