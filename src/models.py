import torch
import config
import torch.nn as nn
from torch.nn.utils import spectral_norm
from layers import (
    CBN,
    SelfAttention,
    ResBlockUp,
    ResBlockDown,
    ResBlock,
    BidirectionalLSTM,
)


class AlphabetLSTM(nn.Module):
    """
    Converts a word into a vectorized format for input to generator
    Args:
    num_tokens: Number of unique tokens in vocabulary (26 in our case)
    emb_size: Size of embedding required
    hidden_dim: Size of output required
    padding: Padding index in vocabulary (0 in our case)
    """

    # num_tokens = 27, 26 chars + 1 <pad>
    def __init__(self, num_tokens, emb_size, hidden_dim, n_layers, padding):
        super(AlphabetLSTM, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size

        self.embedding_layer = nn.Embedding(num_tokens, emb_size, padding_idx=padding)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
        )

    def forward(self, x):
        embedding = self.embedding_layer(x)
        out, _ = self.lstm(embedding)

        return out[-1, :, :]


class GeneratorNetwork(nn.Module):
    """
    Generates the fake handwriting samples (hopefully)
    Args:
    noise_len: Length of the input noise vector
    chunks: Number of chunks to make of the noise vector
    embedding_dim: Size of the embedding from the AlphabetLSTM
    batch_size: Batch size
    """

    def __init__(self, noise_len, chunks, embedding_dim, cbn_mlp_dim, batch_size):
        super(GeneratorNetwork, self).__init__()

        self.split_len = noise_len // chunks
        self.concat_len = self.split_len + embedding_dim
        self.batch_size = batch_size
        self.dense1 = spectral_norm(nn.Linear(self.split_len, 1024))
        # self.dropout = nn.Dropout(p=0.5)

        # inp, cbn_in, emb_size, cbn_hidden, batch_size, out
        self.resblock1 = ResBlockUp(
            256, self.concat_len, cbn_mlp_dim, 1, batch_size, 256
        )
        self.resblock2 = ResBlockUp(
            256, self.concat_len, cbn_mlp_dim, 1, batch_size, 128
        )
        self.resblock3 = ResBlockUp(
            128, self.concat_len, cbn_mlp_dim, 1, batch_size, 128
        )
        self.resblock4 = ResBlockUp(
            128, self.concat_len, cbn_mlp_dim, 1, batch_size, 64
        )
        self.resblock5 = ResBlockUp(64, self.concat_len, cbn_mlp_dim, 1, batch_size, 32)
        self.resblock6 = ResBlockUp(32, self.concat_len, cbn_mlp_dim, 1, batch_size, 16)
        self.resblock7 = ResBlockUp(16, self.concat_len, cbn_mlp_dim, 1, batch_size, 16)
        self.resblocks = [
            self.resblock1,
            self.resblock2,
            self.resblock3,
            self.resblock4,
            self.resblock5,
            self.resblock6,
            self.resblock7,
        ]

        self.self_attention = SelfAttention(64)
        self.penultimate_activation = nn.ReLU()
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels=16, out_channels=1, kernel_size=3, padding=1, bias=False
            )
        )
        self.bn = nn.BatchNorm2d(1)
        self.final_activation = nn.Tanh()

    def forward(self, noise, embedding):
        noise_splits = torch.split(noise, split_size_or_sections=self.split_len, dim=1)
        out = self.dense1(noise_splits[0]).view((self.batch_size, 256, 1, 4))
        for i in range(len(self.resblocks)):
            if i == 4:
                out, _ = self.self_attention(out)
            out = self.resblocks[i](noise_splits[i + 1], embedding, out)
        out = self.penultimate_activation(out)
        return self.final_activation(self.bn(self.conv(out)))


class DiscriminatorNetwork(nn.Module):
    """
    Discriminator, what else you want me to say
    """

    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.resblock1 = ResBlockDown(1, 16)
        self.resblock2 = ResBlockDown(16, 16)
        self.resblock3 = ResBlockDown(16, 32)
        self.resblock4 = ResBlockDown(32, 64)
        self.resblock5 = ResBlockDown(64, 128)
        self.resblock6 = ResBlockDown(128, 128)
        self.resblock7 = ResBlockDown(128, 256)
        self.resblock8 = ResBlock(256, 256)

        self.resdownblocks = [
            self.resblock1,
            self.resblock2,
            self.resblock3,
            self.resblock4,
            self.resblock5,
            self.resblock6,
            self.resblock7,
        ]

        self.self_attention = SelfAttention(32)
        self.global_sum_pooling = nn.LPPool2d(norm_type=1, kernel_size=(1, 4))
        self.dense = spectral_norm(nn.Linear(256, 1))

    def forward(self, x):
        for i, resblock in enumerate(self.resdownblocks):
            if i == 3:
                x, _ = self.self_attention(x)
            x = resblock(x)
        x = self.resblock8(x)
        x = self.global_sum_pooling(x)
        x = x.view((x.shape[0], 256))

        return self.dense(x).view([x.shape[0]])


# TODO: Mention output size after every conv operation as a comment
class R(nn.Module):
    """ScrabbleGAN's OCR Network"""

    def __init__(self):
        super(R, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 2]
        # ks_w = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 0, 0]
        # ps_w = [1, 1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 2, 2]
        # ss_w = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()
        nh = 1024

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))

            cnn.add_module("relu{0}".format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, (2, 2)))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module("pooling{0}".format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module("pooling{0}".format(4), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, config.NUM_TOKENS, nh)
        )
        self.softmax = nn.LogSoftmax(dim=2)

    # Deal with nan/inf losses
    # self.register_backward_hook(self.backward_hook)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.shape
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)

        return self.softmax(output)

    # def backward_hook(self, module, grad_input, grad_output):
    #     for g in grad_input:
    #         g[g != g] = 0  # replace nan/inf with zero
