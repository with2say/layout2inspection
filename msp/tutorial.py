from utils import *
from dataset import *
from model import *
from polytoimage import *


def main(dataset_kwargs={}, n_epoch=200, model_kwargs={}):
    # input params
    n_samples = dataset_kwargs['n_samples']
    n_channels = dataset_kwargs['n_channels']
    range_shape = dataset_kwargs['range_shape']
    range_polygon = dataset_kwargs['range_polygon']
    n_positions = dataset_kwargs['n_positions']
    n_outputs = dataset_kwargs['n_outputs']

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

    model = PolygonRegressor(layer)
    # model.load_from_checkpoint(checkpoint_path)

    trainer = get_trainer(n_epoch)
    trainer.fit(model, data_module)
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    y_true, y_pred = get_predictions(trainer, model, data_module)
    print(np.shape(y_true), np.shape(y_pred))
    evaluate_regression(y_true, y_pred)
    plot_true_vs_predicted(y_true, y_pred)


if __name__ == '__main__':
    dataset_kwargs = {
        'n_samples': 20,
        'n_channels': 2,
        'range_shape': [1, 3],
        'range_polygon': [3, 5],
        'n_positions': 2,
        'n_outputs': 1,
    }
    
    model_kwargs = {
        'polygon_heads': 8, 'polygon_dimension_per_head': 2, 'polygon_layers': 3, 
        'spatial_output_height': 16, 'spatial_output_width': 16, 
        'shape_output_channels': [32, 64, 128],
        'fc_dimensions': 32, 'fc_layers': 1, 'fc_use_batchnorm': False,
    }
    
    main(
        dataset_kwargs=dataset_kwargs,
        n_epoch=3,
        model_kwargs=model_kwargs,
    )    