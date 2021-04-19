import config
import models
import torch
import numpy as np

from train_rec import decode
from dataset import get_dataloader


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def test():
    rec = models.R().to(device)
    checkpoint = torch.load(f"{config.OUT_DIR}/rec_checkpoint.pt")
    rec.load_state_dict(checkpoint["model"])

    _, loader = get_dataloader()

    rec.eval()
    ximgs, xlabels, _ = next(iter(loader))
    ximgs = ximgs.to(device)
    inf_out = rec(ximgs[1, :, :, :].reshape((1, 1, 128, 512)))
    print(f"Network Output: f{decode(torch.argmax(inf_out, dim=2).cpu().numpy(), 0)}")
    print(f"Ground Truth: {xlabels[1]}")


if __name__ == "__main__":
    test()
