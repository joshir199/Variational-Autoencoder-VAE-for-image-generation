import torch
import torch.nn as nn

# Decoder or Generative Network which generates the pixel values based on learned
# or given hidden/latent causes
class DecoderClass(nn.Module):

    def __init__(self, latent_dim=32):
        super(DecoderClass, self).__init__()
        self.latent_dim = latent_dim

        self.fc_shared = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3)
        )

        self.fc_feature = nn.Linear(512, 128*7*7)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )       # 28x28

        # get normalised pixel values of 28x28 size
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, x):

        fc = self.fc_shared(x)
        flatten = self.fc_feature(fc)

        h = self.relu(flatten)

        h = h.view(-1, 128, 7, 7) # reshape the flatten vector to 2D image

        h1 = self.deconv1(h)
        h2 = self.deconv2(h1)
        h3 = self.deconv3(h2)
        h4 = self.deconv4(h3)
        recons = torch.sigmoid(h4)

        return recons

