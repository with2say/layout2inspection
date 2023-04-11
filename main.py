import numpy as np
from dataset import *
from model import *
import seaborn as sns
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
    plt.show()


if __name__ == '__main__':

    # input params
    num_samples = 1000000
    sequence_range = [1, 3]
    polygon_range = [3, 5]
    num_position = 2

    # hyper params
    data, targets = generate_data(num_samples, sequence_range, polygon_range)
    print('target mse:', np.var(targets))

    data_module = PolygonAreaDataModule(data, targets, batch_size=256, val_split=0.05, test_split=0.05, num_workers=4)
    print(np.shape(targets))

    # Create the model
    layer = Polygons2Area(d_model=64,
                          nhead=16,
                          num_layers=16,
                          num_position=num_position,
                          num_polygons=np.max(polygon_range),
                          dropout=0.01)

    # d_model = num_position
    # nhead = 2
    # num_layers = 2
    # dim_feedforward = 4
    # layer = HierarchicalTransformer(d_model=num_position,
    #                                 nhead=nhead,
    #                                 num_layers=num_layers,
    #                                 dim_feedforward=dim_feedforward)

    model = PolygonRegressor(layer)

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)

    y_true, y_pred = get_predictions(trainer, model, data_module)
    plot_true_vs_predicted(y_true, y_pred)


