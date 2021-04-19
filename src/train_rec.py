import config
import models
import pickle
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
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(f"{config.RUNS_DIR}/rec_log")


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


def train(epochs, checkpoint_interval=1):
    rec = models.R().to(device)
    rec_optim = optim.Adam(
        rec.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS
    )

    ctoi_file = open(f"{config.BASE_DIR}/src/ctoi.txt", "rb")
    encoding_dict = pickle.load(ctoi_file)
    ctoi_file.close()
    # print(encoding_dict)

    _, trainloader = get_dataloader()
    ctc_criterion = nn.CTCLoss(zero_infinity=True)

    losses = []
    count = 1

    for epoch in range(epochs):
        epoch_loss = []

        for batch in tqdm(trainloader):
            imgs, labels, lens = batch
            imgs = imgs.to(device)
            labels = preprocess_labels(labels, encoding_dict).transpose(0, 1).to(device)
            lens = torch.LongTensor(lens).to(device)

            rec_optim.zero_grad()

            out = rec(imgs)
            loss = compute_ctc_loss(ctc_criterion, out, labels, lens)
            epoch_loss.append(loss.item())
            loss.backward()

            try:
                for name, param in rec.named_parameters():
                    writer.add_scalar(f"rec gradnorm {name}", param.grad.norm(), count)
            except:
                print("Histogram error in Recognizer")

            rec_optim.step()
            count += 1

        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        writer.add_scalar("Loss", mean_loss, epoch)
        print(f"Epoch {epoch}, Loss: {mean_loss}")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": rec.state_dict(),
                    "opt": rec_optim.state_dict(),
                    "loss": mean_loss,
                },
                f"{config.OUT_DIR}/rec_checkpoint.pt",
            )

    print("Training finished")

    rec.eval()
    ximgs, xlabels, _ = next(iter(trainloader))
    ximgs = ximgs.to(device)
    writer.add_image("test_image", ximgs[1, :, :, :])
    inf_out = rec(ximgs[1, :, :, :].reshape((1, 1, 128, 512)))
    print(f"Network Output: f{decode(torch.argmax(inf_out, dim=2).cpu().numpy(), 0)}")
    print(f"Ground Truth: {xlabels[1]}")


if __name__ == "__main__":
    train(10)
