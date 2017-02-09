import argparse
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def hls_select(img, thresh=(0, 255)):
    """
    Thresholds the S-channel of HLS using using exclusive lower bound (>) and inclusive upper (<=)
    :param img:
    :param thresh:
    :return:
    """
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def show_hls(image_file, visualize=False, save_example=False):
    # Read in an image, you can also try test1.jpg or test4.jpg
    image = mpimg.imread(image_file)
    hls_binary = hls_select(image, thresh=(25, 80))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=24)
    ax2.imshow(hls_binary, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "hls_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HLS S-channel threshold for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize S-channel")
    parser.add_argument("-save", action="store_true", help="Save S-channel image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    show_hls("./test_images/straight_lines2.jpg", visualize, save_examples)
