import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, n_channels, n_shapes, n_polygons, n_positions)
        batch_size, n_channels, n_shapes, n_polygons, _ = x.shape
        x = x.view(batch_size, n_channels, n_shapes, n_polygons, -1)

        position = torch.arange(0, n_polygons, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pos_enc = torch.zeros(n_polygons, self.d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_channels, n_shapes, 1, 1)

        x = x * pos_enc
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
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model

        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: (batch_size, n_channels, n_shapes, n_polygons, d_model)
        batch_size, n_channels, n_shapes, n_polygons, d_model = x.shape
        x = x.view(-1, x.size(3), x.size(4))  # Shape: (batch_size * n_channels * n_shapes, n_polygons, d_model)
        x = x.permute(1, 0, 2)  # Shape: (n_polygons, batch_size * n_channels * n_shapes, d_model)
        
        attn_output, _ = self.self_attention(x, x, x)  # Self-attention
        x = x + attn_output
        x = x.permute(1, 0, 2)  # Shape: (batch_size * n_channels * n_shapes, n_polygons, d_model)

        ff_output = self.positionwise_feedforward(x)
        x = x + ff_output

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
        # print(x.shape)
        x = x.view(batch_size, n_channels, -1)
        # print(x.shape)
        x = self.fc(nn.ReLU()(x))
        # print(x.shape)
        x = x.view(batch_size, n_channels, self.h, self.w)
        return x
    
    # # 어텐션 리턴값에 LayerNorm 추가
    # attention_output = nn.LayerNorm(512)(input + attention_output)

    # # 2개의 FC 레이어 적용
    # fc1 = nn.Linear(512, 2048)
    # fc2 = nn.Linear(2048, 512)
    # fc_output = fc2(nn.ReLU()(fc1(attention_output)))

    # # 출력 shape: (batch_size, n_channels, image_height, image_width)
    # output = fc_output.view(-1, 64, 8, 8)


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


class MultiShapeEmbedding(nn.Module):
    def __init__(self, n_positions, n_polygons, n_shapes, n_channels, n_outputs, d_model, nhead, out_h, out_w):
        super().__init__()
        out_channels = [32, 64, 128]
        self.positional_embedding = PositionalEmbedding(d_model)
        self.polygon_transformer_embedding = PolygonEmbedding(d_model, nhead)
        self.spatial_embedding = SpatialEmbedding(n_shapes, n_positions, out_h, out_w)
        self.shape_embedding = ShapeEmbedding(n_channels, out_channels)
        self.fc = nn.Linear(out_channels[-1], n_outputs)

    def forward(self, x):
        x = self.positional_embedding(x)
        x = self.polygon_transformer_embedding(x)
        x = self.spatial_embedding(x)
        x = self.shape_embedding(x)
        x = self.fc(x)
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
    num_layers = 3
    out_h = 32
    out_w = 32
    
    # MultiShapeEmbedding 객체 생성
    multi_shape_embedding = MultiShapeEmbedding(n_positions, n_polygons, n_shapes, n_channels, n_outputs,
                                                d_model, nhead, out_h, out_w)

    # 무작위 데이터셋 생성
    batch_size = 2
    input_data = torch.randn(batch_size, n_channels, n_shapes, n_polygons, n_positions)

    # MultiShapeEmbedding 실행
    output = multi_shape_embedding(input_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()