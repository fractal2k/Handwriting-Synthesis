import config
import models
import torch
import numpy as np
from tqdm import tqdm

from train_rec import decode
from dataset import get_dataloader
from utils import preprocess_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test():
    rec = models.R().to(device)
    checkpoint = torch.load(f"{config.OUT_DIR}/checkpoint.pt")
    rec.load_state_dict(checkpoint["rec"])

    _, loader = get_dataloader()

    rec.eval()
    ximgs, xlabels, _ = next(iter(loader))
    ximgs = ximgs.to(device)
    inf_out = rec(ximgs[1, :, :, :].reshape((1, 1, 128, 512)))
    print(f"Network Output: f{decode(torch.argmax(inf_out, dim=2).cpu().numpy(), 0)}")
    print(f"Ground Truth: {xlabels[1]}")


if __name__ == "__main__":
    test()
