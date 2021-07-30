import sys
import config
import models
import torch
import pickle
import argparse
import numpy as np
import torchvision
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from inference import inference

# from lookahead import Lookahead
# from augment import DiffAugment
from losses import dis_criterion, gen_criterion, compute_ctc_loss
from utils import imshow, preprocess_labels, generate_noise, clip_norm

from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("true", "t", "yes", "y"):
        return True
    elif v.lower() in ("false", "f", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log", help="Name of the log file for this training run.")
parser.add_argument(
    "-e", "--epochs", default=1, type=int, help="Number of training epochs"
)
parser.add_argument(
    "-c",
    "--checkpoint",
    default=False,
    nargs="?",
    type=str2bool,
    help="Initialize models from last checkpoint",
)
args = parser.parse_args()

# Define global constants
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1")
dis_stats = {"std": 0, "vars": []}
current_log = args.log
writer = SummaryWriter(f"{config.RUNS_DIR}/{current_log}")
policy = "color,translation,cutout"


def init_models():
    """Returns initialized models"""
    lm = models.AlphabetLSTM(
        config.NUM_TOKENS,
        config.EMBEDDING_SIZE,
        config.Z_LEN // 2,
        config.NUM_LAYERS,
        config.PADDING_IDX,
    ).to(device)
    gen = models.GeneratorNetwork(
        config.Z_LEN,
        config.CHUNKS,
        config.EMBEDDING_SIZE,
        config.CBN_MLP_DIM,
        config.BATCH_SIZE,
    ).to(device)
    dis = models.DiscriminatorNetwork().to(device)
    rec = models.R().to(device2)

    return lm, gen, dis, rec


def init_optim(lm, gen, dis, rec):
    """Returns initialized Adam optimizers"""
    lm_opt = optim.Adam(lm.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    gen_opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    dis_opt = optim.Adam(dis.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    rec_opt = optim.Adam(rec.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    return lm_opt, gen_opt, dis_opt, rec_opt


def dis_std(grad):
    """Custom hook function for last layer of recognizer"""
    dis_stats["std"] = np.sqrt(np.mean(dis_stats["vars"]))

    return grad * dis_stats["std"] / grad.std()


def train(epochs=1, from_checkpoint=False, checkpoint_interval=1):
    """Runs training for specified number of epochs"""
    lm, gen, dis, rec = init_models()
    lm_opt, gen_opt, dis_opt, rec_opt = init_optim(lm, gen, dis, rec)

    ctoi_file = open(f"{config.BASE_DIR}/src/ctoi.txt", "rb")
    encoding_dict = pickle.load(ctoi_file)
    ctoi_file.close()

    _, trainloader = get_dataloader()
    stddev = 1
    ctc_criterion = nn.CTCLoss(zero_infinity=True)

    if from_checkpoint:
        point = torch.load(f"{config.OUT_DIR}/checkpoint.pt")
        lm.load_state_dict(point["lm"])
        gen.load_state_dict(point["gen"])
        dis.load_state_dict(point["dis"])
        rec.load_state_dict(point["rec"])
        lm_opt.load_state_dict(point["lm_opt"])
        gen_opt.load_state_dict(point["gen_opt"])
        dis_opt.load_state_dict(point["dis_opt"])
        rec_opt.load_state_dict(point["rec_opt"])
        stddev = point["stddev"]

    gen_losses = []
    dis_losses = []
    rec_losses = []
    count = 1

    for epoch in range(epochs):
        gen_loss_epoch = []
        dis_loss_epoch = []
        rec_loss_epoch = []

        for batch in tqdm(trainloader):
            noise = torch.distributions.normal.Normal(0, stddev)
            stddev -= 0.00001

            imgs, labels, lens = batch
            imgs = imgs.to(device)
            labels = preprocess_labels(labels, encoding_dict).to(device)
            ctc_labels = labels.transpose(0, 1).to(device)
            lens = torch.LongTensor(lens).to(device)

            # ========= Train Discriminator and R =========
            dis_opt.zero_grad()
            rec_opt.zero_grad()

            z_dis = generate_noise(config.Z_LEN, config.BATCH_SIZE, device)
            emb_dis = lm(labels)
            gen_out_dis = gen(z_dis, emb_dis)
            # Adding noise to discriminator input
            # if epoch >= 100:
            # dis_out_fake = dis(gen_out_dis)
            # dis_out_real = dis(imgs)
            # else:
            dis_out_fake = dis(gen_out_dis + noise.sample(gen_out_dis.shape).to(device))
            dis_out_real = dis(imgs + noise.sample(imgs.shape).to(device))

            rec_out_dis = rec(imgs.to(device2))
            dis_loss = dis_criterion(dis_out_fake, dis_out_real)
            rec_loss = compute_ctc_loss(ctc_criterion, rec_out_dis, ctc_labels, lens)
            dis_loss_epoch.append(dis_loss.detach().cpu().numpy())
            rec_loss_epoch.append(rec_loss.detach().cpu().numpy())

            dis_loss.backward()
            rec_loss.backward()

            dis_opt.step()
            rec_opt.step()

            # Creating Histograms of weights
            try:
                for name, param in dis.named_parameters():
                    writer.add_scalar(f"dis gradnorm {name}", param.grad.norm(), count)
            except:
                print("Histogram error in Discriminator")

            try:
                for name, param in rec.named_parameters():
                    writer.add_scalar(f"rec gradnorm {name}", param.grad.norm(), count)
            except:
                print("Histogram error in Recognizer")

            # ========= Train Generator =========
            dis_handles = []
            rec_handles = []
            dis_stats["vars"] = []

            for n, p in rec.named_parameters():
                if n == "rnn.1.out.bias":
                    rec_handles.append(p.register_hook(lambda grad: dis_std(grad)))
                else:
                    rec_handles.append(
                        p.register_hook(
                            lambda grad: grad * dis_stats["std"] / grad.std()
                        )
                    )

            for n, p in dis.named_parameters():
                dis_handles.append(
                    p.register_hook(
                        lambda grad: dis_stats["vars"].append(grad.var().item())
                    )
                )

            lm_opt.zero_grad()
            gen_opt.zero_grad()

            z = generate_noise(config.Z_LEN, config.BATCH_SIZE, device)
            emb = lm(labels)
            gen_out = gen(z, emb)
            rec_out = rec(gen_out.to(device2)).to(device)
            # dis_out = dis(gen_out)
            dis_out = dis(gen_out + noise.sample(gen_out.shape).to(device))

            ctc = compute_ctc_loss(ctc_criterion, rec_out, ctc_labels, lens)
            gen_loss = gen_criterion(dis_out, ctc)
            gen_loss_epoch.append(gen_loss.detach().cpu().numpy())

            gen_loss.backward()

            gen_opt.step()
            lm_opt.step()

            for handle in dis_handles:
                handle.remove()
            for handle in rec_handles:
                handle.remove()

            try:
                for name, param in gen.named_parameters():
                    writer.add_scalar(f"gen gradnorm {name}", param.grad.norm(), count)
            except:
                print("Histogram error in Generator")

            try:
                for name, param in lm.named_parameters():
                    writer.add_scalar(f"lm gradnorm {name}", param.grad.norm(), count)
            except:
                print("Histogram error in Language Model")

            writer.add_scalar("noise stddev", stddev, count)

            count += 1

        # Printing epoch details
        gen_epoch = np.mean(gen_loss_epoch)
        dis_epoch = np.mean(dis_loss_epoch)
        rec_epoch = np.mean(rec_loss_epoch)
        gen_losses.append(gen_epoch)
        dis_losses.append(dis_epoch)
        rec_losses.append(rec_epoch)
        print(
            f"Epoch: {epoch}, Discriminator Loss: {dis_epoch}, Generator Loss: {gen_epoch}, R loss: {rec_epoch}"
        )
        writer.add_scalar("Discriminator loss", dis_epoch, epoch)
        writer.add_scalar("Gererator loss", gen_epoch, epoch)
        writer.add_scalar("R loss", rec_epoch, epoch)

        # Creating model checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "stddev": stddev,
                    "lm": lm.state_dict(),
                    "gen": gen.state_dict(),
                    "dis": dis.state_dict(),
                    "rec": rec.state_dict(),
                    "lm_opt": lm_opt.state_dict(),
                    "gen_opt": gen_opt.state_dict(),
                    "dis_opt": dis_opt.state_dict(),
                    "rec_opt": rec_opt.state_dict(),
                    "dis_loss": dis_epoch,
                    "gen_loss": gen_epoch,
                    "rec_loss": rec_epoch,
                    "log_file": current_log,
                },
                f"{config.OUT_DIR}/checkpoint.pt",
            )
        inference("amit", str(epoch))

    print("Training Finished")


if __name__ == "__main__":
    train(epochs=args.epochs, from_checkpoint=args.checkpoint)
