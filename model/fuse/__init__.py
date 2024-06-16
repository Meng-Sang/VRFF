import torch

from configs import DEVICE
from .UFusion.backbone.UFusion import UFusion


class LazyUFusion(torch.nn.Module):
    def __init__(self):
        super(LazyUFusion, self).__init__()
        self.ufusion = None

    def load_ufusion(self):
        self.ufusion = UFusion(device=DEVICE)
        self.ufusion.load_state_dict(torch.load(f"model/fuse/UFusion/weights/model.pth"))
        self.ufusion = self.ufusion.eval().to(DEVICE)

    def forward(self, ir_image, vi_image):
        if self.ufusion is None:
            self.load_ufusion()
        return self.ufusion(ir_image, vi_image)
