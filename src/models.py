import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from layers import CBN, SelfAttention, ResBlockUp, ResBlockDown, ResBlock


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


class GatedConvolutions(nn.Module):
    """Gated Convolutional Encoder"""

    def __init__(self, in_channels=1):
        super(GatedConvolutions, self).__init__()

        # Big version
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2, 4))
        self.gate1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.gate2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(2, 4))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        # Smol version
        # self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3)
        # self.conv2 = nn.Conv2d(4, 8, kernel_size=(2,4))
        # self.gate1 = nn.Conv2d(8, 8, kernel_size=3, padding = 1)
        # self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        # self.gate2 = nn.Conv2d(16, 16, kernel_size=3, padding = 1)
        # self.conv4 = nn.Conv2d(16, 24, kernel_size=(2,4))
        # self.conv5 = nn.Conv2d(24, 32, kernel_size=3)
        # self.conv6 = nn.Conv2d(32, 64, kernel_size=3)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.gate1(x)) * x
        x = torch.tanh(self.conv3(x))
        x = torch.sigmoid(self.gate2(x)) * x
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        # x = torch.tanh(self.conv6(x))

        return x


class R(nn.Module):
    """Auxiliary "OCR" model to control the generator output"""

    def __init__(self):
        super(R, self).__init__()

        self.encoder = GatedConvolutions(1)
        self.maxpool = nn.MaxPool2d((120, 1))
        # self.maxpool = nn.MaxPool2d((118, 1))

        # Big version
        self.lstm1 = nn.LSTM(
            input_size=500,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(128, 128)
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.linear2 = nn.Linear(256, 27)
        self.softmax = nn.LogSoftmax(dim=1)

        # Smol version doesn't work pls fix later
        # self.lstm1 = nn.LSTM(498, 50, num_layers=1, bidirectional=True)
        # self.dense = nn.Linear(100, 100)
        # self.lstm2 = nn.LSTM(100, 26, num_layers=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.maxpool(x)
        x = x.squeeze(2)
        x, _ = self.lstm1(x)
        bs, sl, hs = x.shape
        x = x.reshape((sl * bs, hs))
        x = self.linear1(x)
        x = x.reshape(bs, sl, -1)
        x, _ = self.lstm2(x)
        bs, sl, hs = x.shape
        x = x.reshape((sl * bs, hs))
        x = self.linear2(x)
        x = x.view(bs, sl, -1)
        x = self.softmax(x).transpose(0, 1)

        return x
