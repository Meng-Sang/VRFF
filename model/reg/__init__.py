import torch

from configs import DEVICE
from .MatchFormer.config.defaultmf import default_cfg
from .MatchFormer.matchformer import MatchFormer


class LazyMatchFormer(torch.nn.Module):
    def __init__(self, config=default_cfg):
        super(LazyMatchFormer, self).__init__()
        self.config = config
        self.matchformer = None

    def load_match_former(self):
        self.matchformer = MatchFormer(config=self.config, device=DEVICE)
        self.matchformer.load_state_dict(torch.load(r"model/reg/MatchFormer/weights/model.ckpt"),
                                         strict=False)
        self.matchformer = self.matchformer.eval().to(DEVICE)

    def forward(self, ir_image, vi_image):
        if self.matchformer is None:
            self.load_match_former()
        return self.matchformer(ir_image, vi_image)
