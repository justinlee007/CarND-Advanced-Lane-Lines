import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration


def apply_transform(img):
    """
    Calculates perspective transform matrix and warps the image
    :param img:
    :return:
    """
    img_shape = (img.shape[1], img.shape[0])
    src, dst = create_warp_mappings(img)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Calculate the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_AREA)
    return warped, M, Minv


def create_warp_mappings(image):
    """
    Creates source and destination coordinates for transforms operation
    :param image:
    :return:
    """
    shape = image.shape
    top_vertical_offset = 90
    bottom_vertical_offset = 50
    horizontal_offset = 120
    mid_vertical = shape[0] / 2
    mid_horizontal = shape[1] / 2
    bottom_left = [horizontal_offset, (shape[0] - bottom_vertical_offset)]
    middle_left = [(mid_horizontal - horizontal_offset), (mid_vertical + top_vertical_offset)]
    middle_right = [(mid_horizontal + horizontal_offset), (mid_vertical + top_vertical_offset)]
    bottom_right = [(shape[1] - horizontal_offset), (shape[0] - bottom_vertical_offset)]
    src = np.float32([middle_right, bottom_right, bottom_left, middle_left])
    dst = np.float32([[shape[1], 0], [shape[1], shape[0]], [0, shape[0]], [0, 0]])
    return src, dst


def show_warp(image_file, visualize=False, save_example=False):
    # Read in an image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Undistort
    dict = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, dict)
    # Get mappings to plot on image
    src, dst = create_warp_mappings(img)
    # Warp image
    warped, M, Minv = apply_transform(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(img)
    for p in src:
        ax1.plot(p[0], p[1], "ro")
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(warped)
    for p in dst:
        ax2.plot(p[0], p[1], "ro")
    ax2.set_title("Warped Image", fontsize=24)
    if (visualize):
        plt.show(block=True)
    if (save_example):
        save_file_name = "warped_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perspective transform for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize transform")
    parser.add_argument("-save", action="store_true", help="Save transform image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    show_warp("./test_images/test2.jpg", visualize, save_examples)
