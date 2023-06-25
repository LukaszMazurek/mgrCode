import numpy as np
import laspy as lp
import open3d as o3d
import pandas as pd
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.under_sampling import NearMiss, RandomUnderSampler
import os



if __name__ == '__main__':

    xyz = np.genfromtxt('datasets/1019_100898/y_pred_spherical.csv', delimiter=',')

    x = [x[1] for x in xyz[1:]]
    y = [x[2] for x in xyz[1:]]
    z = [x[3] for x in xyz[1:]]


    point_data_pred = np.stack([x, y, z], axis=0).transpose((1, 0))

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data_pred)
    o3d.visualization.draw_geometries([geom], window_name='Interpolation')

