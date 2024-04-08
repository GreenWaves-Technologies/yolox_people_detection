import os
from PIL import Image
from tqdm import tqdm

coco_val_path = "/home/marco/Dataset/coco/val2017/"
ppm_coco_val_path = "/home/marco/Dataset/coco/val2017ppm/"
os.makedirs(ppm_coco_val_path, exist_ok=True)

for file in tqdm(os.listdir(coco_val_path)):
    img = Image.open(os.path.join(coco_val_path, file))
    img = img.resize((320, 240))
    filename = os.path.splitext(file)[0]
    out_file_path = os.path.join(ppm_coco_val_path, f"{filename}.ppm")
    img.save(out_file_path)
