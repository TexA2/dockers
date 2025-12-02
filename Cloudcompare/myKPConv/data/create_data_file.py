import pandas as pd
import numpy as np
import scipy
import json
import os
from pypcd4 import PointCloud


labels = {'ground':'1000',
          'car':'1001',
          'track':'1002',
          'buildings':'1003',
          'tree':'1004',
          'unclassified':'1005'}


dir_training_name = "training"
dir_visual_name = "visual"
dir_testing_name = "testing"



def folder_create(folder_path):
    """
    Создает папку, если она не существует
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"✅ Папка '{folder_path}' создана")
        else:
            print(f"ℹ️ Папка '{folder_path}' уже существует")
    except Exception as e:
        print(f"❌ Ошибка при создании папки '{folder_path}': {e}")
        return False


def create_xyz_label_file(cloud_path: str, json_data: str, filename: str) -> np.ndarray:
    with open(json_data) as f:
        data = json.load(f)

    df = pd.DataFrame(data['objects'])
    selected_columns = df[['key', 'classTitle']]

    #получим ключи принадлежащие лейблу
    label_name_key = {}

    for obj in data['objects']:
        class_title = obj['classTitle']
        key = obj['key']

        if class_title not in label_name_key:
            label_name_key[class_title] = []

        label_name_key[class_title].append(key)

    #удалим дубликаты из ground
    # result_ground = []

    
    # for key, indeces in label_name_indices.items():
    #     if key == 'ground':
    #         result_ground.extend(indeces)
    #         continue
        
    #     result_ground = list(set(result_ground) - set(indeces))


    #считываем .pcd файл
    #берем поля x y z
    #intensity пропускаем
    cloud = PointCloud.from_path(cloud_path)
    print(cloud.fields)
    pcnp = cloud.numpy(("x","y","z"))
    print(pcnp.size / 3)
    total_points = int(pcnp.size / 3)


        #получим все точки чтобы в дальнейшем найти землю
    result_ground = []
    result_ground = list(range(total_points))
        

    #получим все индексы по одному лейблу
    label_name_indices = {}

    for obj in data['figures']:
        for class_title, key_list in label_name_key.items():
            if obj['objectKey'] in key_list:

                if class_title not in label_name_indices:
                    label_name_indices[class_title] = []

                label_name_indices[class_title].extend(obj['geometry']['indices'])
                result_ground = list(set(result_ground) - set(label_name_indices[class_title]))
                print(f"{obj['objectKey']}   result_ground = {len(result_ground)}")

    label_name_indices['ground'] = result_ground

    # Проверим индексы
    total_indices = 0
    for class_title, indices in label_name_indices.items():
        print(f"Класс '{class_title}': {len(indices)} точек")
        total_indices += len(indices)


    print(f"total = {total_indices}")
    color_points = np.zeros((int(pcnp.size / 3), 4))

    #генерируем файл
    file_name = filename + ".xyz_label_conf"
    with open(dir_training_name +'/'+file_name, 'w') as f:
        for class_title, indeces in label_name_indices.items():
            print(class_title)
            for i in indeces:
                x, y, z = pcnp[i]
                color_points[i] = [x, y, z, labels[class_title]]
                f.write(f"{x: .2f} {y: .2f} {z: .2f} {labels[class_title]} \n")

    return color_points



# #попытаемся визуализировать 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from visualize_point import visualize_point_cloud

color_map = {
      1000: (1, 1, 1),             #ground
      1001: (1, 0, 0),             #car
      1002: (1, 1, 0),             #track
      1003: (0, 0, 1),             #buildings
      1004: (0, 1, 0),             #tree
      1005: (0.5, 0.5, 0.5)        #unclassified
    }




def create_label_file_from_directory(directory_cloud: str, directory_json: str):
    """
    In :
        directory_cloud: str  Путь до директории c облаком
        directory_json: str  Путь до директории c разметкой
    """
    for filename in os.listdir(directory_cloud):
        if filename.endswith(".pcd"):
            cloud_path = os.path.join(directory_cloud, filename)
            json_name = filename + ".json"
            json_path = os.path.join(directory_json, json_name)

            #create_xyz_label_file(cloud_path, json_path, filename)

            #визуализация
            color_points = create_xyz_label_file(cloud_path, json_path, filename)
            
            colors = [color_map[label] for label in color_points[:,3]]
            points =  color_points[:,:3]

            fig = visualize_point_cloud(points, colors)
            fig.write_html(dir_visual_name +'/'+filename + ".html")


folder_create(dir_training_name)
folder_create(dir_visual_name)
folder_create(dir_testing_name)

create_label_file_from_directory("", "NewAnn/")


