import numpy as np
import cv2
from torch.utils.data import DataLoader, random_split, Dataset

import pytorch_lightning as pl


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def shoelace_formula(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def generate_data(num_samples, seq_length_range, polygon_length_range):
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


def draw_polygons(image, polygons, color=(0, 255, 0), thickness=1):
    for polygon in polygons:
        points = polygon.astype(np.int32).reshape((-1, 1, 2))
        image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        image = cv2.fillPoly(image, [points], color=color)  # Fill the polygon
    return image


def display_polygons(polygons, scale=100, wait_time=3):
    image = np.zeros((scale, scale, 3), dtype=np.uint8)
    scaled_polygons = [p * scale for p in polygons if np.any(p)]  # Scale the polygons
    drawn_image = draw_polygons(image, scaled_polygons)

    cv2.imshow('Polygons', drawn_image)
    while cv2.waitKey(wait_time) != ord('q'):  # Press 'q' to close the window
        pass
    cv2.destroyAllWindows()


class PolygonAreaDataModule(pl.LightningDataModule):
    def __init__(self, data, targets, batch_size=32, val_split=0.2, test_split=0.1, num_workers=3):
        super().__init__()
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = CustomDataset(self.data, self.targets)
        num_val = int(len(dataset) * self.val_split)
        num_test = int(len(dataset) * self.test_split)
        num_train = len(dataset) - num_val - num_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [num_train, num_val, num_test])

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
    polygon_length_range = (3, 5)
    data, targets = generate_data(num_samples, seq_length_range, polygon_length_range)
    print(data)
    # Display polygons
    for polygons in data:
        display_polygons(polygons, scale=200, wait_time=1000)


if __name__ == '__main__':
    main()

