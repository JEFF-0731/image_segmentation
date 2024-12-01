# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm
import chardet

# 預定義標籤類別與對應編號
PREDEFINED_CLASSES = {
    "T7H": 0,
    "T8H": 1,
    "T10H": 2,
    "T15H": 3,
    "PH00": 4,
    "PH0": 5,
    "3/32": 6,
    "1/8": 7
}

def read_json_file(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']
    with open(file_path, "r", encoding=detected_encoding, errors="ignore") as f:
        return json.load(f)

def convert_label_json(json_dir, save_dir, classes):
    json_paths = os.listdir(json_dir)
    os.makedirs(save_dir, exist_ok=True)  # 確保目錄存在
    for json_path in tqdm(json_paths, desc="Converting JSON files"):
        json_path_full = os.path.join(json_dir, json_path)
        try:
            json_dict = read_json_file(json_path_full)
            h, w = json_dict['imageHeight'], json_dict['imageWidth']
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error reading file '{json_path}': {e}")
            continue

        # Save txt path
        txt_path = os.path.join(save_dir, json_path.replace('.json', '.txt'))
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as txt_file:
            for shape_dict in json_dict['shapes']:
                try:
                    label = shape_dict['label']
                    label_index = classes.get(label, None)  # 查找標籤對應的編號
                    if label_index is None:
                        print(f"Label '{label}' not in predefined classes.")
                        continue
                    points = shape_dict['points']
                except ValueError as e:
                    print(f"Error processing shape in file '{json_path}': {e}")
                    continue

                points_nor_list = []
                for point in points:
                    points_nor_list.append(point[0] / w)
                    points_nor_list.append(point[1] / h)

                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_str = ' '.join(points_nor_list)

                label_str = str(label_index) + ' ' + points_nor_str + '\n'
                txt_file.writelines(label_str)

def extract_labels(json_dir):
    labels = set()
    json_paths = os.listdir(json_dir)
    print(f"Found {len(json_paths)} JSON files to process.")
    for json_path in json_paths:
        json_path_full = os.path.join(json_dir, json_path)
        try:
            json_dict = read_json_file(json_path_full)
            for shape_dict in json_dict['shapes']:
                labels.add(shape_dict['label'])
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error reading file '{json_path}': {e}")
    print(f"Extracted labels: {labels}")
    return list(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--json-dir', type=str, default='E:/code/image_segmentation/labelme_json_dir/', help='json path dir')
    parser.add_argument('--save-dir', type=str, default='E:/code/image_segmentation/labelme_txt_dir/', help='txt save dir')
    args = parser.parse_args()

    json_dir = args.json_dir
    save_dir = args.save_dir

    # 使用預定義類別
    predefined_classes = PREDEFINED_CLASSES

    # 進行轉換
    convert_label_json(json_dir, save_dir, predefined_classes)
