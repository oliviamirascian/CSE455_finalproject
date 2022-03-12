# Importing Image and ImageFilter module from PIL package
from PIL import Image, ImageFilter
import os

images = os.listdir("../input/grayscaled/")

for image in images:
    im1 = Image.open("../input/grayscaled/" + image)
    im2 = im1.filter(ImageFilter.BoxBlur(3))
    im2.save("../input/box_filter_blurred/" + image)
