import sys
import numpy as np
from PIL import Image, ImageDraw

BOX_SIZE = 7

image = Image.open(sys.argv[1])
boxes = np.fromfile(sys.argv[2], np.float32).reshape(-1, BOX_SIZE)

draw=ImageDraw.Draw(image)
for box in boxes:
    draw.rectangle([(box[0],box[1]),(box[2],box[3])],outline="white")
image.show()