import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



def get_trainer(n_epoch):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

    # 기울기 클리핑 설정
    gradient_clip_val = 1.0

    # 조기 종료 설정
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=20,
        verbose=True,
        mode="min",
    )

    # 체크포인트 저장 설정
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        mode="min",
    )

    # Stochastic Weight Averaging 설정
    swa_callback = StochasticWeightAveraging(
        swa_epoch_start=100,
        swa_lrs=1e-5,
        # verbose=True,
    )

    import os
    # GPU 사용 설정
    if torch.cuda.is_available():
        gpus = 1  # 사용할 GPU 개수를 지정합니다.
    else:
        gpus = None

    if "COLAB_TPU_ADDR" in os.environ:
        tpu_cores = 8  # 사용할 TPU 코어 개수를 지정합니다.
    else:
        tpu_cores = None

    # Trainer 객체 생성
    trainer = pl.Trainer(
        gpus=gpus, 
        tpu_cores=tpu_cores,
        max_epochs=n_epoch,
        # gradient_clip_val=gradient_clip_val,
        log_every_n_steps=30,
        callbacks=[early_stop_callback, checkpoint_callback, swa_callback],
    )
    return trainer


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


# def evaluate_regression(y_true, y_pred):
#     """Calculate regression metrics and visualize the results.
#     Args:
#         y_true (torch.Tensor): True values of the regression targets.
#         y_pred (torch.Tensor): Predicted values of the regression targets.
#         x (torch.Tensor): Input data, used for visualization purposes.
#     """
#     # Calculate regression metrics
#     mse_loss = F.mse_loss(y_pred, y_true)
#     mae_loss = F.l1_loss(y_pred, y_true)
#     r2 = r2_score(y_true.detach().numpy(), y_pred.detach().numpy())
    
#     # Print regression metrics
#     print(f'MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, R2: {r2:.4f}')


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics.
    Args:
        y_true (list): List of true values of the regression targets.
        y_pred (list): List of predicted values of the regression targets.
    Returns:
        mse (float): Mean squared error.
        mae (float): Mean absolute error.
        r2 (float): R-squared score.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Print regression metrics
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    
    return mse, mae, r2
