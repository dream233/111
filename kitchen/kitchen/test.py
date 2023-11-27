import json
import numpy as np
import os

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

def convert_to_coco_json(input_folder, output_json_path, classes_txt_path):
    coco_format = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    category_dict = {}
    image_id = 1
    annotation_id = 1

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

            coco_format["images"].append({
                "id": image_id,
                "file_name": data["imagePath"],
                "height": data["imageHeight"],
                "width": data["imageWidth"]
            })

            annotations, annotation_id = convert_shapes_to_coco_annotations(
                data["shapes"], image_id, category_dict, annotation_id)
            coco_format["annotations"].extend(annotations)

            image_id += 1

    coco_format["categories"] = [{"id": id, "name": name} for name, id in category_dict.items()]

    with open(output_json_path, 'w') as file:
        json.dump(coco_format, file, indent=4)

    with open(classes_txt_path, 'w') as file:
        for name, id in category_dict.items():
            file.write(f'{id}: {name}\n')

# Replace with your folder path containing JSON files, output file path, and classes txt path
input_folder = 'C:\\Users\\22935\Downloads\\111\\OBJ_DET_IMAGES_ALL_labeled_YOLO-20231116T195349Z-001\\OBJ_DET_IMAGES_ALL_labeled_YOLO\\labels'
output_json_path = 'annotation.json'
classes_txt_path = 'class_with_id.txt'

convert_to_coco_json(input_folder, output_json_path, classes_txt_path)

