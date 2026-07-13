import torch
import torch.nn as nn

class ActionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ActionMLP, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Mean Squared Error (MSE) loss."""
        return self.criterion(predictions, targets)

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute MAE, RMSE, and R^2 metrics."""
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets))
            mse = torch.mean((predictions - targets) ** 2)
            rmse = torch.sqrt(mse)

            # Compute R^2
            target_mean = torch.mean(targets, dim=0)
            ss_tot = torch.sum((targets - target_mean) ** 2)
            ss_res = torch.sum((targets - predictions) ** 2)

            # Handle edge case where ss_tot is zero
            if ss_tot == 0:
                r2 = torch.tensor(1.0 if ss_res == 0 else 0.0, device=predictions.device)
            else:
                r2 = 1 - (ss_res / ss_tot)

        return {
            'mae': mae.item(),
            'rmse': rmse.item(),
            'r2': r2.item()
        }
