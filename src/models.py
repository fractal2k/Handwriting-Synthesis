import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class AlphabetLSTM(nn.Module):
    '''
    Converts a word into a vectorized format for input to generator
    Args:
    num_tokens: Number of unique tokens in vocabulary (26 in our case)
    emb_size: Size of embedding required
    hidden_dim: Size of output required
    padding: Padding index in vocabulary (0 in our case)
    '''

    # num_tokens = 27, 26 chars + 1 <pad>
    def __init__(self, num_tokens, emb_size, hidden_dim, n_layers, padding):
        super(AlphabetLSTM, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size

        self.embedding_layer = nn.Embedding(
            num_tokens, emb_size, padding_idx=padding)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_dim,
                            num_layers=n_layers, bidirectional=True)

    def forward(self, x):
        embedding = self.embedding_layer(x)
        out, _ = self.lstm(embedding)

        return out[-1, :, :]


class CBN(nn.Module):
    """Conditional Batch Normalization implementation from: https://github.com/ap229997/Conditional-Batch-Norm/blob/master/model/cbn.py"""

    def __init__(self, lstm_size, emb_size, out_size, batch_size, channels, height, width, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.lstm_size = lstm_size  # size of the lstm emb which is input to MLP
        self.emb_size = emb_size  # size of hidden layer of MLP
        self.out_size = out_size  # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels))
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels))
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
        )

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''

    def create_cbn_input(self, lstm_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels)

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels)

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)
    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''

    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape

        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean) / \
            torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, lstm_emb


class SelfAttention(nn.Module):
    """ Self attention layer implementation from: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out, attention


class ResBlockUp(nn.Module):
    """Upsampling Residual Block"""

    def __init__(self, inp, cbn_in, emb_size, cbn_hidden, batch_size, out):
        super(ResBlockUp, self).__init__()
        # TODO: Remove the height and width arguments from the CBN Prototype
        # TODO: Implement your own Conditional Batch Normalization layer
        self.cbn1 = CBN(cbn_in, emb_size, cbn_hidden, batch_size, 1, 1, 4)
        self.activation = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = spectral_norm(
            nn.Conv2d(inp, out, kernel_size=3, padding=1, bias=False))
        self.cbn2 = CBN(cbn_in, emb_size, cbn_hidden, batch_size, 1, 1, 4)
        self.conv2 = spectral_norm(
            nn.Conv2d(out, out, kernel_size=3, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(inp, out, kernel_size=1))

    def forward(self, noise_chunk, emb, prev):
        emb_cat = torch.cat((emb, noise_chunk), dim=1)
        x, _ = self.cbn1(prev, emb_cat)
        x = self.activation(x)
        x = self.upsamplex2(x)
        x = self.conv1(x)
        x, _ = self.cbn2(x, emb_cat)
        x = self.activation(x)
        x = self.conv2(x)

        return x + self.conv3(self.upsamplex2(prev))


class ResBlockDown(nn.Module):
    """Downsampling Residual Block"""

    def __init__(self, inp, out):
        super(ResBlockDown, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.activation = nn.LeakyReLU()
        self.avgpool2x2 = nn.AvgPool2d(kernel_size=2)
        self.conv1 = spectral_norm(
            nn.Conv2d(inp, out, kernel_size=3, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = spectral_norm(
            nn.Conv2d(out, out, kernel_size=3, padding=1))
        self.conv3 = spectral_norm(
            nn.Conv2d(inp, out, kernel_size=1))

    def forward(self, x):
        initial = torch.zeros_like(x)
        initial = initial.copy_(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.avgpool2x2(x)

        return x + self.avgpool2x2(self.conv3(initial))


class ResBlock(nn.Module):
    """Regular Residual Block"""

    def __init__(self, inp, out):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.activation = nn.LeakyReLU()
        self.conv1 = spectral_norm(
            nn.Conv2d(inp, out, kernel_size=3, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = spectral_norm(
            nn.Conv2d(out, out, kernel_size=3, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(inp, out, kernel_size=1))

    def forward(self, x):
        initial = torch.zeros_like(x)
        initial = initial.copy_(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x + self.conv3(initial)


class GeneratorNetwork(nn.Module):
    '''
    Generates the fake handwriting samples (hopefully)
    Args:
    noise_len: Length of the input noise vector
    chunks: Number of chunks to make of the noise vector
    embedding_dim: Size of the embedding from the AlphabetLSTM
    batch_size: Batch size
    '''

    def __init__(self, noise_len, chunks, embedding_dim, cbn_mlp_dim, batch_size):
        super(GeneratorNetwork, self).__init__()

        self.split_len = (noise_len // chunks)
        self.concat_len = self.split_len + embedding_dim
        self.batch_size = batch_size
        self.dense1 = spectral_norm(nn.Linear(self.split_len, 1024))

        # inp, cbn_in, emb_size, cbn_hidden, batch_size, out
        self.resblock1 = ResBlockUp(
            256, self.concat_len, cbn_mlp_dim, 1, batch_size, 256)
        self.resblock2 = ResBlockUp(
            256, self.concat_len, cbn_mlp_dim, 1, batch_size, 128)
        self.resblock3 = ResBlockUp(
            128, self.concat_len, cbn_mlp_dim, 1, batch_size, 128)
        self.resblock4 = ResBlockUp(
            128, self.concat_len, cbn_mlp_dim, 1, batch_size, 64)
        self.resblock5 = ResBlockUp(
            64, self.concat_len, cbn_mlp_dim, 1, batch_size, 32)
        self.resblock6 = ResBlockUp(
            32, self.concat_len, cbn_mlp_dim, 1, batch_size, 16)
        self.resblock7 = ResBlockUp(
            16, self.concat_len, cbn_mlp_dim, 1, batch_size, 16)
        self.resblocks = [self.resblock1, self.resblock2, self.resblock3,
                          self.resblock4, self.resblock5, self.resblock6,
                          self.resblock7]

        self.self_attention = SelfAttention(64)
        self.penultimate_activation = nn.ReLU()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, bias=False))
        self.bn = nn.BatchNorm2d(1)
        self.final_activation = nn.Tanh()

    def forward(self, noise, embedding):
        noise_splits = torch.split(
            noise, split_size_or_sections=self.split_len, dim=1)
        out = self.dense1(noise_splits[0]).view((self.batch_size, 256, 1, 4))
        for i in range(len(self.resblocks)):
            if i == 4:
                out, _ = self.self_attention(out)
            out = self.resblocks[i](noise_splits[i + 1], embedding, out)

        out = self.penultimate_activation(out)
        return self.final_activation(self.bn(self.conv(out)))


class DiscriminatorNetwork(nn.Module):
    '''
    Discriminator, what else you want me to say
    '''

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

        self.resdownblocks = [self.resblock1, self.resblock2, self.resblock3,
                              self.resblock4, self.resblock5, self.resblock6,
                              self.resblock7]

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
        self.lstm1 = nn.LSTM(input_size=500, hidden_size=64,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(128, 128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128,
                             num_layers=1, bidirectional=True, batch_first=True)
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
