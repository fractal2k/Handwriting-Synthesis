import config
import models
import torch
import numpy as np
import torchvision
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from lookahead import Lookahead
from dataset import get_dataloader
from losses import compute_ctc_loss
from utils import preprocess_labels

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def decode(arr, blank):
    """Decodes OCR network output
    Args:
    :arr: - Argmax'd output of the network
    :blank: - Blank token index
    """
    res = ""
    for i in range(len(arr)):
        if i == len(arr) - 1:
            if arr[i] != arr[i - 1]:
                res += chr(arr[i][0] + 96)
        elif arr[i] != arr[i + 1]:
            if arr[i] == blank:
                res += "-"
            else:
                res += chr(arr[i][0] + 96)
        else:
            continue

    return res


def train(epochs):
    rec = models.R().to(device)
    rec_optim = Lookahead(optim.Adam(rec.parameters()), la_steps=10)

    _, trainloader = get_dataloader()
    ctc_criterion = nn.CTCLoss()

    losses = []

    for epoch in range(epochs):
        epoch_loss = []

        for batch in tqdm(trainloader):
            imgs, labels, lens = batch
            imgs = imgs.to(device)
            labels = preprocess_labels(labels).transpose(0, 1).to(device)
            lens = torch.LongTensor(lens).to(device)

            rec_optim.zero_grad()

            out = rec(imgs)
            loss = compute_ctc_loss(ctc_criterion, out, labels, lens)
            epoch_loss.append(loss.item())
            loss.backward()
            rec_optim.step()

        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        print(f"Epoch {epoch}, Loss: {mean_loss}")

    print("Training finished")

    rec.eval()
    ximgs, xlabels, _ = next(iter(trainloader))
    ximgs = ximgs.to(device)
    inf_out = rec(ximgs[1, :, :, :].reshape((1, 1, 128, 512)))
    print(f"Network Output: f{decode(torch.argmax(inf_out, dim=2).cpu().numpy(), 0)}")
    print(f"Ground Truth: {xlabels[1]}")


if __name__ == "__main__":
    train(100)
