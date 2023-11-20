import os
import json
import base64
import io
import numpy as np
from PIL import Image
import re
import argparse

def read_name_file(name_path):
    with open(name_path, "r", encoding="utf-8") as name_file:
        # names = [name.strip() for name in name_file]
        names = {name.strip(): i  for i, name in enumerate(name_file)}
        
    return names

def convert_coor(size, xy):
    dw, dh = size
    x, y = xy
    return x / dw, y / dh

def convert(file, txt_name=None):
    # print(file)
    if txt_name is None:
        txt_name = file.rstrip(".json") + ".txt"
        
        if not os.path.exists(os.path.dirname(txt_name)):
            os.makedirs(os.path.dirname(txt_name))
            
        rec_re = re.compile("rectangle_[0-9]+")
        poly_re = re.compile("polygon_[0-9]+")

    names = read_name_file('classes.txt')
    # with open(file, 'r', encoding="utf-8") as f:
    #     Image.fromarray(img_b64_to_arr(json.loads(f.read())['imageData'])).save(txt_name.replace('txt', 'png'))
    
    with open(file, "r", encoding="utf-8") as txt_file:
        js = json.loads(txt_file.read())

        with open(txt_name, "w", encoding="utf-8") as txt_outfile:
                
            for item in js["annotations"]:
                if len(item["segmentation"]) == 0:
                    continue
                
                label = item["category_id"]
                # cls = [str(i) for i, name in enumerate(names) if label == name][0]
                cls = names[label]

                height, width = js["images"]["height"], js["images"]["width"]

                if rec_re.match(item["id"]):
                    if len(item["segmentation"]) != 2:
                        continue
                    
                    x1, y1 = item["segmentation"][0]
                    x2, y2 = item["segmentation"][1]
                    x1_coor, y1_coor = convert_coor((width, height), [x1, y1])
                    x2_coor, y2_coor = convert_coor((width, height), [x2, y2])
                    txt_outfile.write(str(cls) + " " + str(x1_coor) + " " + str(y1_coor)
                                       + " " + str(x2_coor) + " " + str(y1_coor)
                                        + " " + str(x2_coor) + " " + str(y2_coor)
                                         + " " + str(x1_coor) + " " + str(y2_coor) + "\n")
                    
                elif poly_re.match(item["id"]):
                
                    for idx, pt in enumerate(item["segmentation"]):
                        if idx == 0:
                            txt_outfile.write(str(cls))
                            
                        x, y = pt
                        bb = convert_coor((width, height), [x, y])
                        txt_outfile.write(" " + " ".join([str(a) for a in bb]))

                    txt_outfile.write("\n")

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    return Image.open(f)

def img_data_to_arr(img_data):
    return np.array(img_data_to_pil(img_data))

def img_b64_to_arr(img_b64):
    return img_data_to_arr(base64.b64decode(img_b64))

def main():
    print(read_name_file("classes.txt"))
    # parser = argparse.ArgumentParser(description="Convert JSON to TXT")
    # parser.add_argument('--input', type=str, help="Path to the input JSON file", required=True)
    # parser.add_argument('--output', type=str, help="Path to the output TXT file", default=None)
    
    # args = parser.parse_args()
    
    # convert(args.input, args.output)

if __name__ == "__main__":
    main()