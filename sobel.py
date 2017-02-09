import argparse
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    """
    Applies Sobel x or y, then takes an absolute value and applies a threshold.
    :param img:
    :param orient: Derivative in 'x' or 'y'
    :param thresh_min:
    :param thresh_max:
    :return: A mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def show_sobel(image_file, visualize=False, save_example=False):
    # Read in an image and grayscale it
    image = mpimg.imread(image_file)

    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(grad_binary, cmap="gray")
    ax2.set_title("Thresholded Gradient", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "sobel_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sobel threshold for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize threshold")
    parser.add_argument("-save", action="store_true", help="Save threshold image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    show_sobel("./test_images/test3.jpg", visualize, save_examples)
