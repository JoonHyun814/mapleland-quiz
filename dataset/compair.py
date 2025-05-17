import glob
import json

path_list = glob.glob("dataset/all_images/*")

with open("inference/database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)
    database_idx = {v:k for k,v in database_idx.items()}


for path in path_list:
    image_name = path.split("\\")[-1][:-4]
    print(database_idx.pop(image_name))

print(database_idx.keys())