import torch
import config
import models
import torchvision
import argparse
import numpy as np
from utils import imshow, preprocess_labels, generate_noise

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--inp', help='Inference input')
# args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_inference():
    """Returns initialized models"""
    lm = models.AlphabetLSTM(config.NUM_TOKENS,
                             config.EMBEDDING_SIZE,
                             config.Z_LEN // 2,
                             config.NUM_LAYERS,
                             config.PADDING_IDX).to(device)
    gen = models.GeneratorNetwork(config.Z_LEN,
                                  config.CHUNKS,
                                  config.EMBEDDING_SIZE,
                                  config.CBN_MLP_DIM,
                                  config.BATCH_SIZE).to(device)
    return lm, gen


def inference(inp, filename):
    lm, gen = init_inference()
    checkpoint = torch.load(
        '/home/sujayrokade/hsynthesis/src/out/checkpoint.pt')
    # print(
    #     f'Checkpoint Details:\n Trained for: {checkpoint["epoch"]} epochs, Final Generator loss: {checkpoint["gen_loss"]}, Log File: {checkpoint["log_file"]}')
    lm.load_state_dict(checkpoint['lm'])
    gen.load_state_dict(checkpoint['gen'])

    test = preprocess_labels([inp] * config.BATCH_SIZE)
    with torch.no_grad():
        lm.eval()
        gen.eval()
        zin = generate_noise(config.Z_LEN, config.BATCH_SIZE, device)
        gin = lm(test.to(device))
        gout = gen(zin, gin)
        tgrid = torchvision.utils.make_grid(gout.detach().cpu(),
                                            nrow=4)
        imshow(tgrid, f'/home/sujayrokade/hsynthesis/src/out/{filename}.png')

    # print('Inference Finished. Check "out" directory for inference.png')


if __name__ == '__main__':
    inference('sujay')
