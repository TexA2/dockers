import pandas as pd
import numpy as np
import scipy

import open3d as o3d
from pypcd4 import PointCloud
from tqdm import tqdm
import os

from typing import List, Dict, Tuple, Any




class PointCloudDataset(object):
    def __init__(self,data_path: str,grouping_method: str,scales: List,label_map: Dict):

          self.data_path = data_path
          self.grouping_method = grouping_method
          self.scales = scales

          self.label_map = label_map
          self.label_map = {v: k for k, v in self.label_map.items()}

          self.feature_names = ['x', 'y', 'z', 'eigenvals_sum', 'linearity', 'planarity', 'change_of_curvature',
                                 'scattering', 'omnivariance', 'anisotropy', 'eigenentropy','verticality_0','verticality_2',
                                 'height_diff','num_of_points', 'eigenvals_sum_scale2', 'linearity_scale2', 'planarity_scale2',
                                 'change_of_curvature_scale2','scattering_scale2', 'omnivariance_scale2', 'anisotropy_scale2',
                                 'eigenentropy_scale2','verticality_0_scale2','verticality_2_scale2','height_diff_scale2',
                                 'num_of_points_scale2','eigenvals_sum_scale3', 'linearity_scale3', 'planarity_scale3', 'change_of_curvature_scale3',
                                 'scattering_scale3', 'omnivariance_scale3', 'anisotropy_scale3', 'eigenentropy_scale3',
                                 'verticality_0_scale3','verticality_2_scale3','height_diff_scale3','num_of_points_scale3','label','scene_id']

    def read_points_from_file(self, filename: str) -> Tuple[np.ndarray,np.ndarray]:

        data = np.loadtxt(filename)
        points = data[:, 0:3]  # (x, y, z)
        labels = data[:, 3]    # label
        return points, labels

    def load_from_directory(self, directory: str) -> Tuple[List[np.ndarray],List[np.ndarray]]:

        all_points = []
        all_labels = []
        for filename in os.listdir(directory):
            if filename.startswith("cloud") and filename.endswith(".xyz_label_conf"):
              file_path = os.path.join(directory, filename)
              points, labels = self.read_points_from_file(file_path)
              all_points.append(points)
              all_labels.append(labels)

        return all_points, all_labels

    def create_kdtree(self, points :np.ndarray)-> Tuple[o3d.geometry.PointCloud,o3d.geometry.KDTreeFlann]:

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)

        return pcd, tree


    def radius_search(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index : int, radius: float) -> np.ndarray:
        _, indices, _ = tree.search_radius_vector_3d(pcd.points[query_index], radius)
        return np.array(pcd.points)[indices]

    def get_eugen_stats(self, neighbourhood_points: np.ndarray) -> Tuple[float, ...]:
        """
        In :
            neighbourhood_points: np.ndarray  Облако соседних точек найденных  radius_search
        Out :
            feautes: Tuple[float, ...]  -> признаки вычисленные по данному облаку точек
        """

        # Шаг 1. Найдем собственные значения
        covar = np.cov(neighbourhood_points.T)
        assert covar.shape == (3, 3)

        eigenvals, eigenvecs = scipy.linalg.eigh(covar)  # ascending order of eigenvals
        eigenvals, eigenvecs = np.real(eigenvals), np.real(eigenvecs)
        eigenvals, eigenvecs = eigenvals[::-1], eigenvecs[::-1] # descending order of eigenvals
        eigenvals = np.clip(eigenvals, 0, None)

        lambda1,lambda2,lambda3 = eigenvals
        lambda1 = lambda1 + 1e-6

        # Шаг 2. Найдем собственные значения

        # sum of eigenvalues
        sum_of_eigenvalues = lambda1 + lambda2 + lambda3

        # Linearity
        linearity = (lambda1 - lambda2) / lambda1

        # Planarity
        planarity = (lambda2 - lambda3) / lambda1

        # scattering
        scattering = lambda3 / lambda1

        # omnivariance
        omnivariance = (lambda1 * lambda2 * lambda3) ** (1/3.0)

        # anisotropy
        anisotropy = (lambda1 - lambda3) / lambda1

        # eigentropy
        eigenentropy =  -sum([(l / sum_of_eigenvalues) * np.log(l / sum_of_eigenvalues + 1e-6) for l in [lambda1,lambda2,lambda3]])

        verticality_0 = 1 - np.abs(np.dot(np.array([0, 0, 1]), eigenvecs[0]))
        verticality_2 = 1 - np.abs(np.dot(np.array([0, 0, 1]), eigenvecs[2]))

        height_diff = np.max(neighbourhood_points[:,2]) - np.min(neighbourhood_points[:,2])

        num_of_points = neighbourhood_points.shape[0]

        # change of curvative
        change_of_curvature = lambda3 / sum_of_eigenvalues

        return sum_of_eigenvalues, linearity, planarity, change_of_curvature, \
        scattering, omnivariance, anisotropy, eigenentropy,verticality_0,verticality_2,height_diff,num_of_points

    def create_dataset(self) -> pd.DataFrame:
      """
        In :

        Out :
            dataframe : pd.DataFrame Датафрейм с данными, согласно названиям колонок из self.feature_names
        """

      # Шаг 1. Загрузка данных всех сцен из указанной директории(self.data_path)
      scenes,scene_labels = self.load_from_directory(self.data_path)
      points_with_features_and_labels = []

      # Шаг 2. Итерирование по сценам
      for scene_id,data in  enumerate(zip(scenes, scene_labels)):

        # Шаг 3. Создание kdtree
        points,labels = data
        pcd,tree = self.create_kdtree(points)

        # Шаг 4. Итерирование по всем точкам из kdtree
        for index,point in tqdm(enumerate(pcd.points),total=len(pcd.points)):

            # Шаг 5. Поиск соседей одним из методов  radius search
            point_data=[]
            point_data.append(point)
            for scale in self.scales:
              neighbourhood_points = self.radius_search(pcd,tree,index,scale)

              if len(neighbourhood_points) < 3:
                  continue

              # Шаг 6. Вычисление признаков
              features = self.get_eugen_stats(neighbourhood_points)
              point_data.append(features)

            point_data = np.concatenate(point_data)

            label = str(int(labels[index]))
            int_label = self.label_map[label]

            # Шаг 7. Заполнение списка описанием точки [x,y,z,features,label,scene_id] - 1x13
            point_with_label = np.append(point_data,int_label)
            point_with_label_and_scene = np.append(point_with_label,scene_id)
            points_with_features_and_labels.append(point_with_label_and_scene)

      #Шаг 8. Формирование DataFrame
      dataframe = pd.DataFrame(points_with_features_and_labels, columns=self.feature_names)
      return dataframe
    