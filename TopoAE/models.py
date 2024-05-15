import torch
import torch.nn as nn
from eclay import ECLay, AlphaECC, RipsECC


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=101, latent_dim=2, hidden_dim=16):
        """
        Args:
            in_dim (int, optional): Dimension of input point cloud. Defaults to 101.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.loss = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Point cloud. Shape: [B, D]

        Returns:
            tuple: (reconstructed point cloud, latent embedding, loss)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = self.loss(x, x_recon)
        return x_recon, z, loss, (loss.item(), 0)


class TopoAutoEncoder(nn.Module):
    def __init__(self, in_dim=101, latent_dim=2, hidden_dim=16,     # AE params
                 interval=[0, 1], steps=64, scale=0.1,              # ECC params
                #  hidden_features=[32],                              # ECLay params
                 lam=0.1):
        """_summary_

        Args:
            in_dim (int, optional): Dimension of input point cloud. Defaults to 101.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
            lam (float, optional): Controls the weight of topological loss. Defaults to .1.
        """
        super().__init__()
        self.lam = lam
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        # self.eclay_1 = ECLay(interval, steps, hidden_features, scale, type="Rips")
        # self.eclay_2 = ECLay(interval, steps, hidden_features, scale, type="Rips")
        # self.eclay_1 = AlphaECC(interval, steps, scale)
        # self.eclay_2 = AlphaECC(interval, steps, scale)
        self.eclay_1 = RipsECC(interval, steps, scale)
        self.eclay_2 = RipsECC(interval, steps, scale)
        self.loss = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): Point cloud. Shape: [B, D]

        Returns:
            torch.Tensor: (reconstructed point cloud, latent embedding, loss)
        """
        # autoencoder
        z = self.encoder(x)
        x_recon = self.decoder(z)
        # topology layers
        x_max_norm = torch.sqrt(x.square().sum(1)).max()
        z_max_norm = torch.sqrt(z.square().sum(1)).max()
        # normalize point cloud
        ecc_1 = self.eclay_1(x/x_max_norm)
        ecc_2 = self.eclay_2(z/z_max_norm)

        recon_loss = self.loss(x, x_recon)
        topo_loss = self.loss(ecc_1, ecc_2)
        loss = recon_loss + self.lam*topo_loss
        return x_recon, z, loss, (recon_loss.item(), self.lam*topo_loss.item())