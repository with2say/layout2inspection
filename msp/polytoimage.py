import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, input_dim,  d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x: (batch_size, n_channels, n_shapes, n_polygons, n_positions)
        batch_size, n_channels, n_shapes, n_polygons, _ = x.shape
        pos_enc = torch.zeros_like(x)
        for i in range(n_polygons):
            for j in range(2):
                pos_enc[:, :, :, i, j] = torch.sin(x[:, :, :, i, j] / (10000 ** ((2 * i + j) / (2 * self.d_model))))
        
        x = x + pos_enc
        x = x.view(batch_size, n_channels, n_shapes, n_polygons, -1)
        x = self.linear(x)
        return x  # Shape: (batch_size, n_channels, n_shapes, n_polygons, d_model)


# # 좌표가 normalization된 벡터 형태가 된다면 따로 처리 필요 없음.
# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, n_polygons, n_positions):
#         super().__init__()
#         self.embedding = nn.Linear(n_positions, d_model)
#         self.n_polygons = n_polygons

#     def forward(self, x):
#         # x: (batch_size, n_channels, n_shapes, n_polygons, n_positions)
#         batch_size, n_channels, n_shapes, _, _ = x.shape
#         x = x.view(batch_size, n_channels, n_shapes, self.n_polygons, -1)
#         x = self.embedding(x)
#         return x  # Shape: (batch_size, n_channels, n_shapes, n_polygons, d_model)


# (batch_size, n_channels, n_shapes, n_polygons, d_model) -> (batch_size, n_channels, n_shapes, d_model) 
# single head
# class PolygonEmbedding(nn.Module):
#     def __init__(self, d_model, nhead, num_layers):
#         super().__init__()
#         self.d_model = d_model

#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         # x: (batch_size, n_channels, n_shapes, n_polygons, d_model)
#         batch_size, n_channels, n_shapes, n_polygons, d_model = x.shape
#         x = x.view(-1, x.size(3), x.size(4))  # Shape: (batch_size * n_channels * n_shapes, n_polygons, d_model)
#         x = x.permute(1, 0, 2)  # Shape: (n_polygons, batch_size * n_channels * n_shapes, d_model)
#         x = self.encoder(x)
#         x = x.mean(dim=0)  # Shape: (batch_size * n_channels * n_shapes, d_model)
#         x = x.view(batch_size, n_channels, n_shapes, -1)  # Shape: (batch_size, n_channel, n_shape, d_model)
#         return x


class PolygonEmbedding(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1):
        super().__init__()
        self.d_model = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        self.multihead_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # x: (batch_size, n_channels, n_shapes, n_polygons, d_model)
        batch_size, n_channels, n_shapes, n_polygons, d_model = x.shape
        x = x.view(-1, n_polygons, d_model)  # Shape: (batch_size * n_channels * n_shapes, n_polygons, d_model)
        # x = x.permute(1, 0, 2)  # Shape: (n_polygons, batch_size * n_channels * n_shapes, d_model)

        for layer in self.multihead_attention_layers:
            attn_output, _ = layer(x, x, x)  # Self-attention
            x = x + attn_output
            x = self.norm1(x)

        # x = x.permute(1, 0, 2)  # Shape: (batch_size * n_channels * n_shapes, n_polygons, d_model)

        ff_output = self.positionwise_feedforward(x)
        x = x + ff_output
        x = self.norm2(x)

        x = x.mean(dim=1)  # Shape: (batch_size * n_channels * n_shapes, d_model)
        x = x.view(batch_size, n_channels, n_shapes, -1)  # Shape: (batch_size, n_channel, n_shape, d_model)
        return x


# (batch_size, n_channels, n_shapes, d_model) -> (batch_size, n_channels, h, w) 
class SpatialEmbedding(nn.Module):
    def __init__(self, n_shapes, d_model, h, w):
        super().__init__()
        self.fc = nn.Linear(n_shapes*d_model, h * w)
        self.h = h
        self.w = w

    def forward(self, x):
        batch_size, n_channels, n_shapes, d_model = x.shape
        x = x.view(batch_size, n_channels, -1)
        x = self.fc(nn.ReLU()(x))
        x = x.view(batch_size, n_channels, self.h, self.w)
        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.maxpool(x)
        return x


class ShapeEmbedding(nn.Module):
    def __init__(self, n_channels, out_channels=[16, 32, 64]):
        super().__init__()
        self.conv_1 = ConvBlock(n_channels, out_channels[0], 3, 1, 1, 2)
        self.conv_2 = ConvBlock(out_channels[0], out_channels[1], 3, 1, 1, 2)
        self.conv_3 = ConvBlock(out_channels[1], out_channels[2], 3, 1, 1, 2)

    def forward(self, x):
        # x: (batch_size, n_channels, h, w)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.mean(dim=[2, 3])  # Global Average Pooling
        return x  # Shape: (batch_size, d_model)


class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, use_batchnorm=False):
        super().__init__()

        # hidden layers 생성
        hidden_layers = []
        for i in range(num_layers):
            layer = nn.Linear(hidden_size if i > 0 else input_size, hidden_size)
            hidden_layers.append(layer)

            if use_batchnorm:
                bn = nn.BatchNorm1d(hidden_size)
                hidden_layers.append(bn)

            hidden_layers.append(nn.ReLU())

        # fully connected layer 생성
        hidden_layers.append(nn.Linear(hidden_size, output_size))

        # Sequential로 묶기
        self.net = nn.Sequential(*hidden_layers)

    def forward(self, x):
        out = self.net(x)
        return out


class MultiShapeEmbedding(nn.Module):
    def __init__(self, 
                 num_positions, num_polygons, num_shapes, num_channels, num_outputs,
                 polygon_dimension_per_head, polygon_heads, polygon_layers, 
                 spatial_output_height, spatial_output_width, shape_output_channels,
                 fc_dimensions, fc_layers, fc_use_batchnorm=False,
                 ):
        super().__init__()
        self.pos_emb = PositionalEmbedding(num_positions, polygon_dimension_per_head * polygon_heads)
        self.poly_emb = PolygonEmbedding(polygon_dimension_per_head * polygon_heads, polygon_heads, polygon_layers)
        self.spatial_emb = SpatialEmbedding(num_shapes, polygon_dimension_per_head * polygon_heads, spatial_output_height, spatial_output_width)
        self.shape_emb = ShapeEmbedding(num_channels, shape_output_channels)
        self.fc_net = FCNet(shape_output_channels[-1], num_outputs, fc_dimensions, fc_layers, fc_use_batchnorm)

    def forward(self, x):
        x = self.pos_emb(x)
        x = self.poly_emb(x)
        x = self.spatial_emb(x)
        x = self.shape_emb(x)
        x = self.fc_net(x)
        return x

def main():
    # MultiShapeEmbedding에 필요한 파라메터
    n_positions = 2
    n_polygons = 5
    n_shapes = 3
    n_channels = 4
    n_outputs = 1
    d_model = 2
    nhead = 2
    n_layers = 3
    out_h = 32
    out_w = 32
    
    # MultiShapeEmbedding 객체 생성
    multi_shape_embedding = MultiShapeEmbedding(n_positions, n_polygons, n_shapes, n_channels, n_outputs,
                                                d_model, nhead, n_layers, out_h, out_w)

    # 무작위 데이터셋 생성
    batch_size = 2
    input_data = torch.randn(batch_size, n_channels, n_shapes, n_polygons, n_positions)

    # MultiShapeEmbedding 실행
    output = multi_shape_embedding(input_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()