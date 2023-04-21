from utils import *
from dataset import *
from model import *
from polytoimage import *


def main(n_samples=10000, n_epoch=200, model_kwargs={}):
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

    data_module = PolygonAreaDataModule(data, targets, batch_size=128, val_split=0.1, test_split=0.1, num_workers=2)

    # MultiShapeEmbedding 객체 생성
    layer = MultiShapeEmbedding(
        n_positions, n_polygons, n_shapes, n_channels, n_outputs,
        **model_kwargs,
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

    trainer = get_trainer(n_epoch)
    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)

    y_true, y_pred = get_predictions(trainer, model, data_module)
    print(np.shape(y_true), np.shape(y_pred))
    evaluate_regression(y_true, y_pred)

    plot_true_vs_predicted(y_true, y_pred)



if __name__ == '__main__':
    model_kwargs = {
        'polygon_heads': 8, 'polygon_dimension_per_head': 2, 'polygon_layers': 3, 
        'spatial_output_height': 16, 'spatial_output_width': 16, 
        'shape_output_channels': [32, 64, 128],
        'fc_dimensions': 32, 'fc_layers': 1, 'fc_use_batchnorm':False,
    }
    
    main(
        n_samples=10, 
        n_epoch=1,
        model_kwargs=model_kwargs,
    )    