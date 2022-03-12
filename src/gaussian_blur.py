# Importing Image and ImageFilter module from PIL package
from PIL import Image, ImageFilter
import os

images = os.listdir("../input/grayscaled/")

for image in images:
    im1 = Image.open("../input/grayscaled/" + image)
    im2 = im1.filter(ImageFilter.GaussianBlur(3))
    im2.save("../input/gaussian_blurred/" + image)
