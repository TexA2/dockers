import os
from tools.visualize_point import visualize_point_cloud
import catboost
from catboost import CatBoostClassifier, Pool
from tools.data_load_scale import PointCloudDataset
import json
from pypcd4 import PointCloud
import numpy as np
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))

cloud_path = os.path.join(script_dir, "data/cloud/cloud.pcd")
setting_path = os.path.join(script_dir, "settings.json")


#JSON PARSER
with open(setting_path) as f:
    config = json.load(f)


labels = config['labels']
color_map = config['color_map']
scales = config['dataset_config']['scales']
method = config['dataset_config']['method']
color_map = {int(k): tuple(v) for k, v in config["color_map"].items()}
label_map = {0:'1000',1:'1001',2:'1002',3:'1003',4:'1004', 5:'1005'}
test_data_path = config['dataset_config']['test_data_path']

test_data_path = os.path.join(script_dir, test_data_path)
print(test_data_path)
######


#Обработка PCD файла
cloud = PointCloud.from_path(cloud_path)
print(cloud.fields)
pc = cloud.numpy(("x","y","z","intensity"))

pcnp = pc[:,:3]
pc_intensity = pc[:,3]


file_name = test_data_path + "/" + "cloud" + ".xyz_label_conf"
print(file_name)
with open(file_name, 'w') as f:
    for point in tqdm(pcnp):
        x, y, z = point
        f.write(f"{x: .2f} {y: .2f} {z: .2f} 1000\n")
####


# Инициализация модели и ее выполнение
model = CatBoostClassifier()

model_path = os.path.join(script_dir, "best_catboost_model_SCALE.cbm")
model.load_model(model_path)

test_dataset = PointCloudDataset(test_data_path,method,scales,label_map)
test_dataframe = test_dataset.create_dataset()

X_test = test_dataframe.drop(columns=["label","scene_id"])
y_test = test_dataframe['label']
y_pred = model.predict(X_test)


points  = test_dataframe.to_numpy()[:,:3]
labels = y_pred.reshape(-1,)
colors = [color_map[label] for label in labels]
####

# Подготовка визуализации в виде html файла
fig = visualize_point_cloud(points, colors)
fig.write_html(script_dir + "/Predict_point.html")
####

# Обработка и сохранение Point Clouds
mask_car = np.isin(labels, [1.0, 2.0])
mask_road = np.isin(labels, 0.0)
mask_static_obj = ~mask_car

points_car = points[mask_car]
points_car_intensity = pc_intensity[mask_car]

points_static_obj = points[mask_static_obj]
points_static_obj_intensity = pc_intensity[mask_static_obj]


point_road = points[mask_road]
intensity_road = pc_intensity[mask_road]

combined_car = np.column_stack((points_car, points_car_intensity))
combined_static_obj = np.column_stack((points_static_obj, points_static_obj_intensity))
combined_road = np.column_stack((point_road, intensity_road))



pc1 = PointCloud.from_points(combined_car, fields=('x', 'y', 'z', 'intensity'),
                            types=(np.float32, np.float32, np.float32, np.float32))


pc2 = PointCloud.from_points(combined_static_obj, fields=('x', 'y', 'z', 'intensity'),
                              types=(np.float32, np.float32, np.float32, np.float32))

pc3 = PointCloud.from_points(combined_road, fields=('x', 'y', 'z', 'intensity'),
                              types=(np.float32, np.float32, np.float32, np.float32))

pc1.save(script_dir + "/CAR_cloud.pcd")
pc2.save(script_dir + "/Static_obj_cloud.pcd")
pc3.save(script_dir + "/Road_cloud.pcd")
