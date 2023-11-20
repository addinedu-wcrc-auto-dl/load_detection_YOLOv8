import os
import glob
from labelme2yolo import convert

def convert_folder(folder_path):
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    for file in json_files:
        convert(file)

convert_folder("../../ref/aihub/data/train/label_east_indoor")
convert_folder("../../ref/aihub/data/train/label_west_indoor")
convert_folder("../../ref/aihub/data/val/label_east_indoor")
convert_folder("../../ref/aihub/data/val/label_west_indoor")