import pandas as pd
import numpy as np
import scipy
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold, StratifiedKFold

import catboost
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN


class Trainer(object):
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, model_config: Dict, n_folds: int):
        """
      In :
          train_data: pd.DataFrame - таблица с данными для тренировки
          test_data: pd.DataFrame - таблица с данными для тестирования
          model_config : Dict - словарь с параметрами модели
          n_folds : int - количество фолдов
        """

        self.model_params = model_config
        self.n_folds = n_folds
        self.train_data = train_data
        self.test_data = test_data
        self.metrics = ['TotalF1:average=Weighted;use_weights=False', 'Accuracy']
        self.folds = self.create_folds()

    def create_folds(self) -> List[List[np.ndarray]]:
        """
        In :
            self.train_data : pd.DataFrame  - тренировочные данные
            self.n_folds : int - количество фолдов
        Out :
            kfold_dset_index : List[List[np.ndarray]] - список пар индексов train_index,val_index для каждого из n фолдов
        """ 
        X = self.train_data.drop(columns=["label","scene_id"])
        y = self.train_data['label']
        groups = self.train_data['scene_id']
        #stratified_kfold = StratifiedGroupKFold(n_splits=self.n_folds)
        stratified_kfold = GroupKFold(n_splits=self.n_folds)
        kfold_dset_index = []
        for train_index, val_index in stratified_kfold.split(X, y,groups):
            kfold_dset_index.append([train_index, val_index])
        return kfold_dset_index
    
    def train_catboost(self, train_pool: catboost.Pool, val_pool: catboost.Pool) -> catboost.CatBoostClassifier:
        """
        In :
            train_pool : catboost.Pool - инициализированный коеструктор для тренировочных данных (относящихся к train_index)
            val_pool : catboost.Pool - инициализированный коеструктор для валидационных данных (относящихся к val_index)
        Out :
            catboost_model : catboost.CatBoostClassifier - обученная модель
        """ 
        catboost_model = CatBoostClassifier(**self.model_params)
        catboost_model.fit(train_pool, eval_set = val_pool, use_best_model=True)
        return catboost_model
    


    def test_catboost(self, test_pool: catboost.Pool, catboost_model: catboost.CatBoostClassifier) -> List[float]:
        """
        In :
            test_pool : catboost.Pool - инициализированный коеструктор для тестовых данных (относящихся к self.test_data)
            catboost_model : catboost.CatBoostClassifier - обученная модель
        Out :
            metric_values : List[float] - значение метрик для комбинации всех деревьев (https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier_eval-metrics ntree_end)
        """ 
        metric_result = catboost_model.eval_metrics(test_pool, self.metrics)
        metric_values = []
        for key in metric_result.keys():
            print(f"{key}={metric_result[key][-1]}")
            metric_values.append(metric_result[key][-1])

        return metric_values

    def fit(self):
        """
        In :
            self.train_data: pd.DataFrame - таблица с данными для тренировки
            self.test_data: pd.DataFrame - таблица с данными для тестирования
            self.folds : List[List[np.ndarray]] - список пар индексов train_index,val_index для каждого из n фолдов 
        Out :
            self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
            self.acc_list : List[float] - список из значений метрики Accuracy на тестовой выборке для каждой из обученных моделей
            self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей
        """
        #Шаг 1 Сформируем X,y X_test, y_test убрав ненужные колонки из pd.DataFrame
        X = self.train_data.drop(columns=["label","scene_id"])
        y = self.train_data["label"]

        X_test =  self.test_data.drop(columns=["label","scene_id"])
        y_test = self.test_data["label"]

        self.f1_list = []
        self.acc_list = []
        self.model_list = []


        #Шаг 3 Проходимся  по  self.folds
        for i, fold_split_index in tqdm(enumerate(self.folds), total=len(self.folds), desc='KFold train-test iterations'):

            #Step 4 create X_train y_train X_val y_val use index folds
            train_index, val_index = fold_split_index
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
##
            labels, counts = np.unique(y_train,return_counts=True)
            upsample_val = 20000
            target = {label: max(count, upsample_val) for label, count in zip(labels, counts)}
            smote = SMOTE(sampling_strategy = target)
            X_train, y_train = smote.fit_resample(X_train, y_train)
##

            X_val, y_val = X.iloc[val_index], y.iloc[val_index]

            #Step 4 inicialize train_pool , val_pool
            train_pool = Pool(data=X_train, label = y_train, has_header=False)
            val_pool = Pool(data=X_val, label = y_val, has_header=False)

            #Step 5 train cat_boost
            catboost_model = self.train_catboost(train_pool,val_pool)

            print(f"{i} iteration")

            #Step 6 inicialize test_pool from X_test y_test
            test_pool = Pool(data=X_test, label = y_test, has_header=False)

            f1, acc = self.test_catboost(test_pool, catboost_model)

            self.f1_list.append(f1)
            self.acc_list.append(acc)
            self.model_list.append(catboost_model)

            del X_train, y_train, X_val, y_val, train_pool, val_pool, test_pool, catboost_model

        # Шаг 7 Выведем среднее и стандартное отклонение полученных по n фолдам метрик
        print('\n================================')
        print(f"F1={np.mean(self.f1_list)} +/- {np.std(self.f1_list)}")
        print(f"Accuracy={np.mean(self.acc_list)} +/- {np.std(self.acc_list)}")


    def return_best(self):
        """
        In :
            self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
            self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей

        Out :
            self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
            self.acc_list : List[float] - список из значений метрики Accuracy на тестовой выборке для каждой из обученных моделей
            self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей
        """
        model_index = self.f1_list.index(max(self.f1_list))
        model = self.model_list[model_index]
        return model
    

    def predict_ensemble(self, X) -> np.ndarray:
        prediction = []
        for model in self.model_list:
            pred = model.predict(X)
            prediction.append(pred)

        return scipy.stats.mode(prediction, axis=0)[0]
