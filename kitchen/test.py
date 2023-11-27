import json
import numpy as np
import os
import random

def convert_shapes_to_coco_annotations(shapes, image_id, category_dict, annotation_id):
    annotations = []
    for shape in shapes:
        label = shape["label"]
        if label not in category_dict:
            category_dict[label] = len(category_dict) + 1

        points = shape["points"]
        np_points = np.array(points)
        min_x, min_y = np.min(np_points, axis=0)
        max_x, max_y = np.max(np_points, axis=0)
        width, height = max_x - min_x, max_y - min_y
        area = np.abs(np.dot(np_points[:, 0], np.roll(np_points[:, 1], 1)) - np.dot(np_points[:, 1], np.roll(np_points[:, 0], 1))) / 2.0

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_dict[label],
            "segmentation": [list(np.concatenate(np_points).flat)],
            "area": area,
            "bbox": [float(min_x), float(min_y), float(width), float(height)],
            "iscrowd": 0
        })

        annotation_id += 1

    return annotations, annotation_id

def split_coco_json(input_folder, test_json_path, trainval_json_path, split_ratio=0.1):
    all_data = []
    category_dict = {}
    image_id = 1
    annotation_id = 1

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            image_info = {
                "id": image_id,
                "file_name": data["imagePath"],
                "height": data["imageHeight"],
                "width": data["imageWidth"]
            }

            annotations, annotation_id = convert_shapes_to_coco_annotations(
                data["shapes"], image_id, category_dict, annotation_id)

            all_data.append((image_info, annotations))
            image_id += 1

    random.shuffle(all_data)
    test_count = int(len(all_data) * split_ratio)
    test_data = all_data[:test_count]
    trainval_data = all_data[test_count:]

    def create_coco_json(data, path):
        coco_format = {
            "images": [item[0] for item in data],
            "type": "instances",
            "annotations": [ann for item in data for ann in item[1]],
            "categories": [{"id": id, "name": name} for name, id in category_dict.items()]
        }
        with open(path, 'w') as file:
            json.dump(coco_format, file, indent=4)

    create_coco_json(test_data, test_json_path)
    create_coco_json(trainval_data, trainval_json_path)

# Replace with your folder path containing JSON files, and output file paths
input_folder = 'C:\\Users\\22935\Downloads\\111\\kitchen\\labels'
output_json_path = 'annotation_all.json'
test_json_path = 'test.json'
trainval_json_path = 'trainval.json'

split_coco_json(input_folder, test_json_path, trainval_json_path)



