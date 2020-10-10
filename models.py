from torch import nn
from torch.autograd import grad
import torch
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, latent_size, multiplier=4, img_size=64, vae=False):
        super(AE, self).__init__()
        self.fm = 4
        self.mp = multiplier
        self.encoder = nn.Sequential(
            nn.Conv2d(3, int(16 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(16 * multiplier),
                      int(32 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(32 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64 * multiplier)),
            nn.ReLU(True),
            # 128 x 128
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False) if img_size>=128 else nn.Identity(),
            nn.BatchNorm2d(int(64 * multiplier)) if img_size>=128 else nn.Identity(),
            nn.ReLU(True) if img_size>=128 else nn.Identity(),
            # 256 x 256
            nn.Conv2d(int(64 * multiplier),
                      int(64 * multiplier), 4, 2, 1, bias=False) if img_size>=256 else nn.Identity(),
            nn.BatchNorm2d(int(64 * multiplier)) if img_size>=256 else nn.Identity(),
            nn.ReLU(True) if img_size>=256 else nn.Identity(),
        )
        if not vae:
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm*self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size),
            )
        else:
            self.linear_enc = nn.Sequential(
                nn.Linear(int(64 * multiplier) * self.fm*self.fm, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),
                nn.Linear(2048, latent_size * 2),
            )

        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, int(64 * multiplier) * self.fm*self.fm),
        )
        self.decoder = nn.Sequential(
            # 128 x 128
            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False) if img_size>=128 else nn.Identity(),
            nn.BatchNorm2d(int(64*multiplier)) if img_size>=128 else nn.Identity(),
            nn.ReLU(True) if img_size>=128 else nn.Identity(),
            # 256 x 256
            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False) if img_size>=256 else nn.Identity(),
            nn.BatchNorm2d(int(64*multiplier)) if img_size>=256 else nn.Identity(),
            nn.ReLU(True) if img_size>=256 else nn.Identity(),

            nn.ConvTranspose2d(int(64*multiplier), int(64 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(64*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(64*multiplier), int(32 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(32*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(32*multiplier), int(16 *
                                                       multiplier), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(16*multiplier)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(16*multiplier),
                               3, 4, 2, 1, bias=False),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        lat_rep = self.feature(x)
        out = self.decode(lat_rep)
        return out

    def feature(self, x):
        lat_rep = self.encoder(x)
        lat_rep = lat_rep.view(lat_rep.size(0), -1)
        lat_rep = self.linear_enc(lat_rep)
        return lat_rep

    def decode(self, x):
        out = self.linear_dec(x)
        out = out.view(out.size(0), int(64 * self.mp), self.fm, self.fm)
        out = self.decoder(out)
        out = torch.tanh(out)
        return out


class Spatial2DAE(nn.Module):
    def __init__(self):
        super(Spatial2DAE, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, padding=5//2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, 2, padding=5//2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, 2, padding=5//2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, 2, padding=5//2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 16, 3, 2, padding=3//2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 128, 3, 2, padding=3//2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2, padding=5//2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=5//2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, 2, padding=5//2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 5, 2, padding=5//2,
                               bias=False, output_padding=1),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x)
        x = torch.tanh(x)
        return x
