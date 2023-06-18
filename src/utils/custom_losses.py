import torch
import torch.nn as nn

class MatrixLoss(nn.Module):
    """
    The MatrixLoss class is a PyTorch Module that calculates the Mean Absolute Error (MAE) 
    between the distance matrices of predicted and target coordinates of atoms in a molecule.
    
    The class uses a function `distance_matrix(x)` which calculates the Euclidean distance 
    matrix of a set of 3D coordinates. The function takes an input tensor of shape (N, 3), 
    where N is the number of atoms, and returns a matrix of shape (N, N) where the entry 
    at (i, j) is the Euclidean distance between atoms i and j.

    The forward method of this class computes the distance matrices of the predicted and 
    target coordinates, and then calculates the MAE between these two matrices.

    This loss function is useful in applications where the absolute difference between 
    the pairwise distances of atoms is important, such as in molecular dynamics and 
    computational chemistry models.
    
    Attributes:
    mae: An instance of PyTorch's L1Loss function, which computes the mean absolute error.

    Methods:
    forward(pred, target): Computes the loss between the predicted and target coordinates.
    """

    def __init__(self):
        super(MatrixLoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        """
        Compute the loss between predicted and target coordinates.
        
        Args:
        pred: Predicted coordinates, a tensor of shape (N, 3).
        target: Target coordinates, a tensor of shape (N, 3).
        
        Returns:
        loss: The mean absolute error loss between the distance matrices of predicted and target coordinates.
        """
        pred_dist_mat = self.distance_matrix(pred)
        target_dist_mat = self.distance_matrix(target)
        return self.mae(pred_dist_mat, target_dist_mat)
    
    @staticmethod
    def distance_matrix(x):
        """
        x: Tensor of shape (N, 3) where N is the number of atoms.
        This function returns a matrix of shape (N, N) where the entry at (i, j) is
        the Euclidean distance between atoms i and j.
        """
        # Expand x along the second dimension
        x = x.unsqueeze(1)

        # Calculate the pairwise distance in one line by taking advantage of broadcasting
        dist = torch.sqrt(torch.sum((x - x.transpose(0, 1)) ** 2, -1)+1e-8)

        return dist
    