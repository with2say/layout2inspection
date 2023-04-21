import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning.pytorch as pl


class PolygonRegressor(pl.LightningModule):
    def __init__(self, layer, lr=1e-3):
        super().__init__()
        self.layer = layer
        self.lr = lr
        self.save_hyperparameters()

    def get_loss(self, batch):
        polygons, areas = batch
        outputs = self.layer(polygons)
        loss = nn.MSELoss()(outputs, areas)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_pred = self.layer(x)
        return y_pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {"optimizer": optimizer} # "lr_scheduler": scheduler, "monitor": "val_loss"
    

class PolygonEmbedding(nn.Module):
    def __init__(self, d_model, num_polygons, num_position):
        super().__init__()
        self.embedding = nn.Linear(num_polygons*num_position, d_model)
        self.num_polygons = num_polygons
        self.num_position = num_position

    def forward(self, x):
        # x: (batch_size, seq_length, num_polygon, num_position)
        batch_size, seq_length, _, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_polygons * self.num_position)
        x = self.embedding(x)
        return x


class Polygons2Area(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_polygons, num_position, dropout=0.5):
        super().__init__()
        self.polygon_embedding = PolygonEmbedding(d_model, num_polygons, num_position)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, num_polygon, num_position)
        x = self.polygon_embedding(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x


class HierarchicalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.polygon_transformer = nn.Transformer(
            d_model, nhead, num_layers, dim_feedforward
        )
        self.sequence_transformer = nn.Transformer(
            d_model, nhead, num_layers, dim_feedforward
        )
        self.fc1 = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, num_sequence, num_polygon, _ = x.size()

        # Process polygons
        x = x.view(batch_size * num_sequence, num_polygon, -1)
        x = self.polygon_transformer(x, x)  # Use the same input for src and tgt
        x = x.mean(dim=1)

        # Process sequences
        x = x.view(batch_size, num_sequence, -1)
        x = self.sequence_transformer(x, x)  # Use the same input for src and tgt
        x = self.fc1(x)

        # Reshape to outputs
        x = torch.mean(x, dim=1)

        return x


def main():
    num_samples = 10
    num_sequence = 3
    num_polygon = 4
    num_position = 2

    polygons = torch.rand(num_samples, num_sequence, num_polygon, num_position)
    areas = torch.rand(num_samples, num_sequence, 1)

    dataset = TensorDataset(polygons, areas)
    dataloader = DataLoader(dataset, batch_size=32)

    d_model = num_position
    nhead = 2
    num_layers = 2
    dim_feedforward = 64
    model = HierarchicalTransformer(d_model, nhead, num_layers, dim_feedforward)
    model = PolygonRegressor(model)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
    