from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self._find_last_conv(model)
        self.activations = None
        self.gradients = None
        self._h1 = self.target_layer.register_forward_hook(self._forward_hook)
        self._h2 = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv(self, m: nn.Module) -> nn.Module:
        last = None
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                last = mod
        if last is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM")
        return last

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, logits: torch.Tensor, class_idx: Optional[int] = None):
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        loss = logits[0, class_idx]
        loss.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap if hooks didn't capture data
            return np.ones((224, 224), dtype=np.float32)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def attention_rollout_vit(model: nn.Module, attn_layer_name: str = "blocks") -> np.ndarray:
    # Placeholder: robust fallback returning a uniform heatmap. Implement proper
    # attention extraction for your specific ViT implementation if needed.
    heat = np.ones((224, 224), dtype=np.float32)
    return heat / heat.max()
