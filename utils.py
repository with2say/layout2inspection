import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def get_predictions(trainer, model, data_module):
    model.eval()
    model.to('cpu')
    y_true = []
    y_pred = []

    predictions = trainer.predict(model, datamodule=data_module)

    for pred, y in predictions:
        y_true.extend(y.view(-1).tolist())
        y_pred.extend(pred.view(-1).tolist())

    return y_true, y_pred


def plot_true_vs_predicted(y_true, y_pred):
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()


def evaluate_regression(y_true, y_pred, x=None):
    """Calculate regression metrics and visualize the results.
    Args:
        y_true (torch.Tensor): True values of the regression targets.
        y_pred (torch.Tensor): Predicted values of the regression targets.
        x (torch.Tensor): Input data, used for visualization purposes.
    """
    # Calculate regression metrics
    mse_loss = F.mse_loss(y_pred, y_true)
    mae_loss = F.l1_loss(y_pred, y_true)
    r2 = r2_score(y_true.detach().numpy(), y_pred.detach().numpy())
    
    # Print regression metrics
    print(f'MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, R2: {r2:.4f}')

