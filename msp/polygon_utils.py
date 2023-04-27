import random
import numpy as np
from shapely.geometry import Polygon
from rtree import index
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon


def compute_overlapping_areas(shapes_1, shapes_2):
    # 데이터를 Shapely 다각형 객체로 변환
    polygons_1 = [Polygon(shape) for shape in shapes_1]
    polygons_2 = [Polygon(shape) for shape in shapes_2]

    overlapping_areas = []
    for polygon_1 in polygons_1:
        for polygon_2 in polygons_2:
            intersection = polygon_1.intersection(polygon_2)
            if intersection.area > 0:
                overlapping_areas.append(intersection.area)

    return overlapping_areas


def create_vertices_random(n_polygons, x_range=(-1, 1), y_range=(-1, 1)):
    return [(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])) for _ in range(n_polygons)]


def create_vertices_centered_on_lines(n_polygons, n_lines=5, x_range=(-1, 1), y_range=(-1, 1)):
    line_positions = np.linspace(x_range[0], x_range[1], n_lines + 2)[1:-1]
    line_idx = random.randint(0, n_lines - 1)
    center_x = line_positions[line_idx]
    center_y = random.uniform(y_range[0], y_range[1])

    angle = np.linspace(0, 2 * np.pi, n_polygons + 1)[:-1]
    radius = random.uniform(0.1, 0.2)
    vertices = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angle]

    return vertices


class PolygonGenerator:
    def __init__(self, check_overlap=True, check_min_distance=False, check_area=False, min_distance=0.1, min_area=0.01, max_area=0.1):
        self.check_overlap = check_overlap
        self.check_min_distance = check_min_distance
        self.check_area = check_area
        self.min_distance = min_distance
        self.min_area = min_area
        self.max_area = max_area
        self.idx = index.Index()
        self.polygons = []

    def create_vertices(self, n_polygons, x_range=(-1, 1), y_range=(-1, 1)):
        return [(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])) for _ in range(n_polygons)]

    def is_overlapping(self, polygon):
        if not self.check_overlap:
            return False
        return any(self.idx.intersection(polygon.bounds))

    def is_below_min_distance(self, polygon):
        if not self.check_min_distance:
            return False
        for i in self.idx.intersection(polygon.bounds):
            existing_polygon = self.polygons[i]
            if polygon.distance(existing_polygon) < self.min_distance:
                return True
        return False

    def is_valid_area(self, polygon):
        if not self.check_area:
            return True
        return self.min_area <= polygon.area <= self.max_area

    def generate_polygon(self, n_polygons, x_range=(-1, 1), y_range=(-1, 1), vertex_generation_method=create_vertices_random, **kwargs):
        while True:
            vertices = vertex_generation_method(n_polygons, x_range=x_range, y_range=y_range, **kwargs)

            polygon = Polygon(vertices)

            if not self.is_overlapping(polygon) and not self.is_below_min_distance(polygon) and self.is_valid_area(polygon):
                self.polygons.append(polygon)
                self.idx.insert(len(self.polygons) - 1, polygon.bounds)
                return polygon

    def generate_polygons(self, n_shapes, n_polygons, x_range=(-1, 1), y_range=(-1, 1), vertex_generation_method=create_vertices_random, **kwargs):
        polygons = []
        for _ in range(n_shapes):
            polygon = self.generate_polygon(n_polygons, x_range, y_range, vertex_generation_method, **kwargs)
            polygons.append(polygon)
        return polygons


def plot_polygons_old(polygons):
    fig, ax = plt.subplots()

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, marker='o')

    plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon
from matplotlib.collections import PatchCollection

def plot_polygons(polygons, x_range=(-1, 1), y_range=(-1, 1)):
    fig, ax = plt.subplots()

    patches = []
    for polygon in polygons:
        patches.append(mpl_Polygon(np.array(polygon.exterior.coords), closed=True))

    p = PatchCollection(patches, alpha=0.4, edgecolor='black', linewidths=2)
    ax.add_collection(p)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('polygons.png')
    plt.show()


pg = PolygonGenerator(check_overlap=True, check_min_distance=True, check_area=True, min_distance=0.1, min_area=0.01, max_area=0.1)
polygons = pg.generate_polygons(10, 4, vertex_generation_method=create_vertices_centered_on_lines)
print(polygons)
plot_polygons(polygons)

