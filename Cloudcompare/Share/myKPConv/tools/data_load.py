import pandas as pd
import numpy as np
import scipy

import open3d as o3d
from pypcd4 import PointCloud
from tqdm import tqdm
import os

from typing import List, Dict, Tuple, Any




class PointCloudDataset(object):
    def __init__(self, data_path: str, grouping_method: str, neighbourhood_th: Any, label_map: Dict):
        """
        data_path: str - путь до папки с данными
        grouping_method : str - метод поиска соседей , ["knn","radius_search",имплементированный вами]
        neighbourhood_th : Any[int,float] - пороговое значение для k - количества соседей или radius - радиуса сферы
        label_map : Dict - словарь {label : index}
        """

        self.data_path = data_path
        self.grouping_method = grouping_method
        self.neighbourhood_th = neighbourhood_th

        self.label_map = label_map
        self.label_map = {v: k for k, v in self.label_map.items()}

        self.feature_names = ['x', 'y', 'z', 'eigenvals_sum', 'linearity', 'planarity', 'change_of_curvature',
                                'scattering', 'omnivariance', 'anisotropy', 'eigenentropy', 'label','scene_id']
        
    def read_points_from_file(self ,filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        In :
            filename: str  Путь до файла с облаком точек
        Out :
            points,labels : Tuple[np.ndarray,np.ndarray] -> массивы точек , лейблов
        """
        
        data = np.loadtxt(filename)
        points = data[:, :3] #(x, y, z)
        labels = data[:, 3]  #labels

        return points, labels        

    def load_from_directory(self, directory: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        In :
            directory: str  Путь до директории с файлами
        Out :
            all_points, all_labels : Tuple[List[np.ndarray],List[np.ndarray]] Набор точек,лейблов для каждой сцены
        """
        all_points = []
        all_labels = []

        for filename in os.listdir(directory):
            if filename.startswith("cloud") and filename.endswith(".xyz_label_conf"):
                file_path = os.path.join(directory, filename)
                points, labels = self.read_points_from_file(file_path)
                all_points.append(points)
                all_labels.append(labels)

        return all_points, all_labels
    

    def create_kdtree(self, points :np.ndarray) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.KDTreeFlann]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)

        return pcd, tree
    
    def knn(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index: int, k: int) -> np.array:
        """
        In :
            pcd: o3d.geometry.PointCloud  Облако точек
            tree : o3d.geometry.KDTreeFlann k - d дерево (https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html)
            query_index : int Индекс точки в дереве для которой нужно найти соседей
            k : int Количество ближайших соседей для поиска
        Out :
            points: np.ndarray  -> найденные точки(включая query_index)
        """
        _, indices, _ = tree.search_knn_vector_3d(pcd.points[query_index], k)

        return np.array(pcd.points)[indices]       


    def radius_search(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index : int, radius: float) -> np.ndarray:
        """
        In :
            pcd: o3d.geometry.PointCloud  Облако точек
            tree : o3d.geometry.KDTreeFlann k - d дерево (https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html)
            query_index : int Индекс точки в дереве для которой нужно найти соседей
            radius : float Радиус сферы , в метрах
        Out :
            points: np.ndarray  -> найденные точки(включая query_index)
        """
        _, indices, _ = tree.search_radius_vector_3d(pcd.points[query_index], radius)
        return np.array(pcd.points)[indices]


    def get_eigen_stats(self, neighbourhood_points: np.ndarray) -> Tuple[float,...]:
        neighbourhood_points = neighbourhood_points.T
        cenred_data = neighbourhood_points - np.mean(neighbourhood_points, axis = 1)[:, None]
        covar = np.cov(cenred_data)
        assert covar.shape == (3,3)

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

        # change of curvative
        change_of_curvature = lambda3 / sum_of_eigenvalues

        return sum_of_eigenvalues, linearity, planarity, change_of_curvature, \
                scattering, omnivariance, anisotropy, eigenentropy
    

    def create_dataset(self) -> pd.DataFrame:
        """
        Out :
            dataframe : pd.DataFrame Датафрейм с данными, согласно названиям колонок из self.feature_names
        """       
       # Шаг 1. Загрузка данных всех сцен из указанной директории(self.data_path)
        scenes,scene_labels = self.load_from_directory(self.data_path)
        points_with_features_and_labels = []

        # Шаг 2. Итерирование по сценам
        for scene_id, data in  enumerate(zip(scenes, scene_labels)):

            # Шаг 3. Создание kdtree
            points,labels = data
            pcd,tree = self.create_kdtree(points)

            # Шаг 4. Итерирование по всем точкам из kdtree
            for index,point in tqdm(enumerate(pcd.points),total=len(pcd.points)):

                # Шаг 5. Поиск соседей одним из методов - knn или radius search
                if self.grouping_method == "knn":
                    neighbourhood_points = self.knn(pcd,tree,index,self.neighbourhood_th)
                if self.grouping_method == "radius_search":
                    neighbourhood_points = self.radius_search(pcd,tree,index,self.neighbourhood_th)

                if len(neighbourhood_points) < 3:
                    continue

                # Шаг 6. Вычисление признаков
                features = self.get_eigen_stats(neighbourhood_points)
                point_data = np.concatenate([point,features])

                label = str(int(labels[index]))
                int_label = self.label_map[label]

                # Шаг 7. Заполнение списка описанием точки [x,y,z,features,label,scene_id] - 1x13
                point_with_label = np.append(point_data,int_label)
                point_with_label_and_scene = np.append(point_with_label,scene_id)
                points_with_features_and_labels.append(point_with_label_and_scene)

        #Шаг 8. Формирование DataFrame
        dataframe = pd.DataFrame(points_with_features_and_labels, columns=self.feature_names)
        return dataframe

        

