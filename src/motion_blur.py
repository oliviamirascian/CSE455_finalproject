
import numpy as np
from PIL import Image, ImageFilter
import os
import cv2

images = os.listdir("../input/original_data/");

# Specify the kernel size.
# The greater the size, the more the motion.
kernel_size = 30

# Create the vertical kernel.
kernel_v = np.zeros((kernel_size, kernel_size))

# Create a copy of the same for creating the horizontal kernel.
kernel_h = np.copy(kernel_v)

# Fill the middle row with ones.
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

# Normalize.
kernel_v /= kernel_size
kernel_h /= kernel_size


# horizontal blur
for image in images:
    im1 = cv2.imread("../input/original_data/" + image)
    # Apply the horizontal blur kernel.
    im2 = cv2.filter2D(im1, -1, kernel_h)
    cv2.imwrite("../input/motion_blurred/" + image, im2)


