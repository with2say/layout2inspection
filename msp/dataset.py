import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import lightning.pytorch as pl


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ShuffleRollingAugmentationDataset(Dataset):
    def __init__(self, data, targets, shuffle_axes=None, rolling_axes=None, p=0.3, pos_min=[0, 0], pos_max=[1, 1]):
        self.data = data
        self.targets = targets
        self.shuffle_axes = shuffle_axes if shuffle_axes is not None else []
        self.rolling_axes = rolling_axes if rolling_axes is not None else []
        self.p = p

        self.pos_min=pos_min
        self.pos_max=pos_max

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        
        if random.random() < self.p:
            for axis in self.shuffle_axes:
                data = self.shuffle(data, axis=axis)
        
        if random.random() < self.p:                
            for axis in self.rolling_axes:
                data = self.roll(data, axis=axis)
        
        if random.random() < self.p:
            data = self.flip_lr(data)
        
        if random.random() < self.p:
            data = self.flip_ud(data)
            
        return data, target

    def shuffle(self, data, axis):
        idx = torch.randperm(data.shape[axis])
        return data.index_select(axis, idx)

    def roll(self, data, axis):
        shift = random.randint(0, data.shape[axis] - 1)
        return torch.roll(data, shifts=shift, dims=axis)
    
    def flip_lr(self, data):
        data[..., 0] = self.pos_max[0] - (data[..., 0] - self.pos_min[0])
        return data

    def flip_ud(self, data):
        data[..., 1] = self.pos_max[1] - (data[..., 1] - self.pos_min[1])
        return data


def shoelace_formula(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def generate_data_with_none_padding(num_samples, seq_length_range, polygon_length_range):
    data = []
    targets = []

    min_seq_len, max_seq_len = seq_length_range
    min_poly_len, max_poly_len = polygon_length_range

    for _ in range(num_samples):
        seq_length = np.random.randint(min_seq_len, max_seq_len + 1)
        sequence = []
        area_sum = 0

        for _ in range(seq_length):
            polygon_length = np.random.randint(min_poly_len, max_poly_len + 1)
            coords = np.random.rand(polygon_length, 2).astype(np.float32)
            padded_coords = list(coords) + [None] * (max_poly_len - polygon_length)
            sequence.append(padded_coords)
            area_sum += shoelace_formula(coords)

        sequence += [None] * (max_seq_len - seq_length)
        data.append(sequence)
        targets.append(area_sum)

    return np.array(data), np.array(targets).reshape(-1, 1).astype(np.float32)


def generate_data_with_negative_padding(num_samples, seq_length_range, polygon_length_range):
    nan_value = -0.1
    data = []
    targets = []

    min_seq_len, max_seq_len = seq_length_range
    min_poly_len, max_poly_len = polygon_length_range

    for _ in range(num_samples):
        seq_length = np.random.randint(min_seq_len, max_seq_len + 1)
        sequence = []
        area_sum = 0

        for _ in range(seq_length):
            polygon_length = np.random.randint(min_poly_len, max_poly_len + 1)
            coords = np.random.rand(polygon_length, 2).astype(np.float32)
            padded_coords = np.pad(coords, [(0, max_poly_len - polygon_length), (0, 0)], mode='constant', constant_values=-1)
            sequence.append(padded_coords)
            area_sum += shoelace_formula(coords)

        for _ in range(max_seq_len - seq_length):
            padded_coords = np.full((max_poly_len, 2), nan_value, dtype=np.float32)
            sequence.append(padded_coords)

        data.append(sequence)
        targets.append(area_sum)

    return np.array(data), np.array(targets).reshape(-1, 1).astype(np.float32)


def generate_dataset(n_samples, n_channels, range_shape, range_polygon):
    # 데이터 및 타겟을 n_channels 만큼 생성합니다.
    datasets = [generate_data_with_negative_padding(n_samples, range_shape, range_polygon) for _ in range(n_channels)]

    # 데이터를 axis=1 기준으로 연결합니다.
    data = np.stack([dataset[0] for dataset in datasets], axis=1)

    # 타겟을 모두 더합니다.
    targets = np.sum([dataset[1] for dataset in datasets], axis=0)

    return data, targets


def draw_polygons(image, polygons, color=(0, 255, 0), thickness=1):
    import cv2
    for polygon in polygons:
        points = polygon.astype(np.int32).reshape((-1, 1, 2))
        image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        image = cv2.fillPoly(image, [points], color=color)  # Fill the polygon
    return image


# def draw_polygons(image, polygons, color=(0, 255, 0), thickness=1):
#     from PIL import Image, ImageDraw
#     img_pil = Image.fromarray(image)
#     draw = ImageDraw.Draw(img_pil)

#     for polygon in polygons:
#         points = polygon.astype(np.int32).reshape((-1, 2)).tolist()
#         # draw.line(points + [points[0]], fill=color, width=thickness)
#         draw.polygon(points, fill=color)

#     return np.array(img_pil)


def display_polygons(polygons, scale=100, wait_time=3):
    image = np.zeros((scale, scale, 3), dtype=np.uint8)
    scaled_polygons = [p * scale for p in polygons if np.any(p)]  # Scale the polygons
    drawn_image = draw_polygons(image, scaled_polygons)

    cv2.imshow('Polygons', drawn_image)
    while cv2.waitKey(wait_time) != ord('q'):  # Press 'q' to close the window
        pass
    cv2.destroyAllWindows()


class PolygonAreaDataModule(pl.LightningDataModule):
    def __init__(self, data, targets, batch_size=32, val_split=0.2, test_split=0.1, aug_prob=0.3, num_workers=3):
        super().__init__()
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.aug_prob = aug_prob
        self.num_workers = num_workers

    def setup(self, stage=None):   
        dataset = ShuffleRollingAugmentationDataset(self.data, self.targets, p=self.aug_prob)
        num_val = int(len(dataset) * self.val_split)
        num_test = int(len(dataset) * self.test_split)
        num_train = len(dataset) - num_val - num_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [num_train, num_val, num_test])
        self.train_dataset.shuffle_axes=2
        self.train_dataset.rolling_axes=3

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def main():
    # Generate data
    num_samples = 10
    seq_length_range = (1, 3)
    polygon_length_range = (2, 3)
    data, targets = generate_data_with_negative_padding(num_samples, seq_length_range, polygon_length_range)
    print(np.shape(data), np.shape(targets))

    # Display polygons
    for polygons in data:
        # print(polygons)
        display_polygons(polygons, scale=200, wait_time=1000)




if __name__ == '__main__':
    main()

