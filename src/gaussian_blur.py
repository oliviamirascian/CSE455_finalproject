# Importing Image and ImageFilter module from PIL package
from PIL import Image, ImageFilter
import os

images = os.listdir("../input/original_data/");

for image in images:
    im1 = Image.open("../input/original_data/" + image);
    im2 = im1.filter(ImageFilter.GaussianBlur(5))
    im2.save("../input/gaussian_blurred/" + image)