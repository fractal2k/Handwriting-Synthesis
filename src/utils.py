import torch
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, filename):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(8, 8))
    inp = (inp + 1) / 2
    inp = inp.numpy().transpose((1, 2, 0))

    plt.imsave(filename, inp)
    plt.close(fig)


def convert_word(word):
    return [ord(c) - 96 for c in word]


def preprocess_labels(labels):
    max_len = len(max(labels, key=len))
    final = []
    for label in labels:
        encoding = convert_word(label) + [0] * (max_len - len(label))
        final.append(encoding)

    return torch.LongTensor(final).transpose(0, 1)


def generate_noise(z_len, batch_size, device):
    return torch.randn((batch_size, z_len)).to(device)


def clip_norm(grad, max_val):
    """PyTorch's clip_grad_norm_ implemented for register hooks"""
    max_val = float(max_val)
    norm = torch.norm(grad.detach(), 2)

    clip_coef = max_val / (norm + 1e-6)

    if clip_coef < 1:
        return grad.detach().mul(clip_coef.to(grad.device))
    else:
        return grad
