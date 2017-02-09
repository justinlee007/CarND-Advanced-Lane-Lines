import argparse
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration
import transform


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


# Applies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Applies Sobel x and y, then computes the direction of the gradient and applies a threshold.
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def combined_sobel(img):
    """
    Applies each of the thresholding functions
    :param img:
    :return:
    """
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=40, thresh_max=255)
    # grady = abs_sobel_thresh(img, orient='y', thresh_min=30, thresh_max=255)
    mag_binary = mag_thresh(img, sobel_kernel=13, mag_thresh=(70, 200))
    # dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.6, 1.4))

    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # combined = np.zeros_like(mag_binary)
    # combined[((gradx == 1) & (grady == 1)) | (mag_binary == 1)] = 1
    # combined[(gradx == 1) | (mag_binary == 1)] = 1
    # combined = cv2.bitwise_or(cv2.bitwise_and(gradx, grady), mag_binary)
    combined = cv2.bitwise_or(gradx, mag_binary)
    return combined


def show_sobel(image_file, visualize=False, save_example=False):
    # Read in an image and grayscale it
    image = mpimg.imread(image_file)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(image)

    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=40, thresh_max=255)

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


def show_mag_dir(image_file, visualize=False, save_example=False):
    # Read in an image
    image = mpimg.imread(image_file)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(image)

    # Run the function
    mag_binary = mag_thresh(image, sobel_kernel=13, mag_thresh=(70, 200))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=24)
    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "mag_dir_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


def show_dir_thresh(image_file, visualize=False, save_example=False):
    # Read in an image
    image = mpimg.imread(image_file)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(image)

    # Run the function
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.6, 1.4))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(dir_binary, cmap="gray")
    ax2.set_title("Thresholded Grad. Dir.", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "dir_thresh_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


def show_combined(image_file, visualize=False, save_example=False):
    # Read in an image
    image = mpimg.imread(image_file)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    image = camera_calibration.undistort_image(image, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(image)

    # Run the function
    combined = combined_sobel(image)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(combined, cmap="gray")
    ax2.set_title("Combined Sobel Thresholds", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "combined_sobel_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sobel threshold for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize threshold")
    parser.add_argument("-save", action="store_true", help="Save threshold image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    # show_combined("./test_images/test1.jpg", visualize, save_examples)
    # show_combined("./test_images/test2.jpg", visualize, save_examples)
    # show_combined("./test_images/test3.jpg", visualize, save_examples)
    show_combined("./test_images/test4.jpg", visualize, save_examples)
    # show_combined("./test_images/test5.jpg", visualize, save_examples)
    # show_combined("./test_images/test6.jpg", visualize, save_examples)
