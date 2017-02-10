import argparse
import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration
import transform


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """
    Applies Sobel x or y, then takes an absolute value and applies a threshold.
    :param img:
    :param orient: Derivative in 'x' or 'y'
    :param thresh_min:
    :param thresh_max:
    :return: A mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    """
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def color_threshold(image, hls_thresh=(0, 255), hsv_thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= hls_thresh[0]) & (s_channel <= hls_thresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= hsv_thresh[0]) & (v_channel <= hsv_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def gaussian_blur(img, kernel=5):
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blur


def show_imaging(image_file, visualize=False, save_example=False):
    image = mpimg.imread(image_file)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(image)

    # Run the function
    sobel_x = abs_sobel_thresh(image, orient='x', thresh=(35, 255))
    sobel_y = abs_sobel_thresh(image, orient='y', thresh=(15, 255))

    sobel_binary = cv2.bitwise_and(sobel_x, sobel_y)

    color_binary = color_threshold(image, hls_thresh=(150, 255), hsv_thresh=(200, 255))

    processed_image = cv2.bitwise_or(sobel_binary, color_binary)

    processed_image = gaussian_blur(processed_image, kernel=9)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(processed_image, cmap="gray")
    ax2.set_title("Processed Image", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "imaging_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imaging methods for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize imaging process")
    parser.add_argument("-save", action="store_true", help="Save processed image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)

    # show_sobel_combined_channel("./test_images/test1.jpg", visualize, save_examples)
    images = glob.glob("./test_images/test*.jpg")
    for file_name in images:
        show_imaging(file_name, visualize, save_examples)
