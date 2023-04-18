from utils import *
from dataset import *
from model import *
from polytoimage import *


def main(n_samples=10000, n_epoch=200):
    # input params
    n_samples = n_samples
    n_channels = 2
    range_shape = [1, 3]
    range_polygon = [3, 5]
    n_positions = 2
    n_outputs = 1

    # checkpoint_path = '/workspaces/layout2inspection/lightning_logs/version_15/checkpoints/epoch=21-step=704.ckpt'

    # addtional params
    n_shapes = np.max(range_shape)
    n_polygons = np.max(range_polygon)

    # hyper params
    # data, targets = generate_data_with_negative_padding(n_samples, range_shape, range_polygon)
    data, targets = generate_dataset(n_samples, n_channels, range_shape, range_polygon)
    print('null mse:', np.var(targets))
    print(data.shape, targets.shape)

    data_module = PolygonAreaDataModule(data, targets, batch_size=512, val_split=0.1, test_split=0.1, num_workers=2)

    # MultiShapeEmbedding 객체 생성
    d_model = 8
    nhead = 2
    num_layers = 4
    out_h = 32
    out_w = 32
    layer = MultiShapeEmbedding(
        n_positions, n_polygons, n_shapes, n_channels, n_outputs,
        d_model, nhead, num_layers, out_h, out_w
    )

    # Create the model
    # layer = Polygons2Area(d_model=64,
    #                       nhead=16,
    #                       num_layers=16,
    #                       num_position=num_position,
    #                       num_polygons=np.max(polygon_range),
    #                       dropout=0.01)

    # d_model = num_position
    # nhead = 2
    # num_layers = 2
    # dim_feedforward = 4
    # layer = HierarchicalTransformer(d_model=num_position,
    #                                 nhead=nhead,
    #                                 num_layers=num_layers,
    #                                 dim_feedforward=dim_feedforward)

    model = PolygonRegressor(layer)
    # model.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(max_epochs=n_epoch)
    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)

    y_true, y_pred = get_predictions(trainer, model, data_module)
    print(np.shape(y_true), np.shape(y_pred))
    plot_true_vs_predicted(y_true, y_pred)


if __name__ == '__main__':
    main(n_samples=10, n_epoch=20)