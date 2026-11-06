"""Grad-CAM 可解释性：生成目标类别在卷积特征上的响应热力图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def overlay_heatmap(input_tensor, cam, mean, std, alpha=0.5):
    """将 CAM 叠加到原图，返回可直接保存的 RGB 数组（0~255）。"""
    t = input_tensor.detach().cpu()
    t = t * torch.tensor(mean).view(3, 1, 1) \
        + torch.tensor(std).view(3, 1, 1)
    image = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()

    cam_resized = F.interpolate(
        torch.from_numpy(cam).float().view(1, 1, *cam.shape),
        size=image.shape[:2], mode="bilinear", align_corners=False,
    ).squeeze().numpy()
    heatmap = plt.get_cmap("jet")(cam_resized)[:, :, :3]

    overlay = (1 - alpha) * image + alpha * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
