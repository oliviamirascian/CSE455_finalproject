import cv2
import os

src_dir = '../input/animefaces'
images = os.listdir(src_dir)
dst_dir = '../input/grayscaled'

for i, img in enumerate(images):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
