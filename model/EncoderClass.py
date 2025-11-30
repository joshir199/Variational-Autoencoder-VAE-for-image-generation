import torch
import torch.nn as nn

# Encoder part is also called inference or recognition model
# It tries to get good approximate distribution over the latent variables
# that could have generated the given input image
# Here, Encoder will output the distribution of latent variable in form of
# its mean and variance of the probability distribution
# both mean and variance are of dimension-d. So, a total of 2d vector as output
class EncoderClass(nn.Module):

    def __init__(self, latent_dim=32):
        # input image = [B, 1, 28, 28]
        super(EncoderClass, self).__init__()
        self.latent_dim = latent_dim

        # avoid pooling layers for smoother gradient flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1) # C*H*W = 128*7*7

        self.fc_shared = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3)
        )

        # Separate linear layer for each latent features
        # so that they learn independently
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Initialisation helps avoid posterior collapse in early training
        nn.init.constant_(self.fc_logvar.weight, 0.0)
        nn.init.constant_(self.fc_logvar.bias, -6.0)  # ≈ σ² = 0.0025 → σ ≈ 0.05


    def forward(self, image):

        # convolution heads
        h1 = self.conv1(image)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        # flattening head
        hf = self.flatten(h4)

        # fully connected head
        h_fc = self.fc_shared(hf)

        # encoder output for mean and log_variance
        mean = self.fc_mean(h_fc)
        log_var = self.fc_logvar(h_fc)

        return mean, log_var

