# import the libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import nnSqueeze

### Functions for STN ###


def init_weight_STN(stn):
    """Initialize the weights/bias with (nearly) identity transformation
    reference: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """
    stn[-5].weight[:, -28 * 28:].data.zero_()
    stn[-5].bias.data.zero_()
    stn[-1].weight.data.zero_()
    stn[-1].bias.data.copy_(torch.tensor([1 - 1e-2, 1e-2,
                            1 - 1e-2], dtype=torch.float))


def convert_Avec_to_A(A_vec):
    """Convert BxM tensor to BxNxN symmetric matrices"""
    """M = N * (N + 1) / 2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 3:
        A_dim = 2
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec

    return A.squeeze()

### Encoder Module ###


class Encoder(nn.Module):
    """Create the Encoder module"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, domain_dim):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # number of hidden layers
        self.output_dim = output_dim  # number of features
        self.dropout = dropout
        self.domain_dim = domain_dim

        # Spatial Transformer Network (STN): provide the model with a stronger inductive bias
        self.fc_stn = nn.Sequential(
            # input layer: 28x28 input image and domain index
            nn.Linear(self.domain_dim + self.input_dim * \
                      self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 3)  # output layer: outputs for STN
        )

        # Encoder: learn domain-invariant features
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=3,
                      stride=2, padding=1),  # returns 14x14
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3,
                      stride=2, padding=1),  # returns 7x7
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3,
                      stride=2, padding=1),  # returns 4x4
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.hidden_dim, self.output_dim, kernel_size=4,
                      stride=1, padding=0),  # returns 1x1
            nn.ReLU(True)
        )

    def stn(self, x, u):
        A_vec = self.fc_stn(torch.cat([u, x.reshape(-1, 28 * 28)], 1))
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        tcos, tsin = evs[:, 0:1, 0:1], evs[:, 1:2, 0:1]

        self.theta_angle = torch.atan2(tsin[:, 0, 0], tcos[:, 0, 0])

        # clockwise rotate theta
        theta_0 = torch.cat([tcos, tsin, tcos * 0], 2)
        theta_1 = torch.cat([-tsin, tcos, tcos * 0], 2)
        theta = torch.cat([theta_0, theta_1], 1)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, u):
        x_align = self.stn(x, u)  # x_align
        features = self.conv(x_align)  # features

        # returns STN output and domain-invariant encodings
        return x_align, features


### Classifier Module ###

class Classifier(nn.Module):
    """Create the Classifier module"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        self.input_dim = input_dim  # number of features
        self.hidden_dim = hidden_dim  # number of hidden layers
        self.output_dim = output_dim  # number of classes

        # Predictor: predict class labels
        self.fc_pred = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
            nnSqueeze(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        predictions = F.log_softmax(self.fc_pred(x), dim=1)  # predictions

        return predictions


### Discriminator Module ###


class Discriminator(nn.Module):
    """Create the Discriminator module"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # number of hidden layers
        self.output_dim = output_dim  # domain_dim

        # Discriminator: predict domain index with domain-invariant encodings
        self.fc_disc = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # returns prediction of domain index
        return self.fc_disc(x)
