import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == '__main__':
    # Read in an image
    image = mpimg.imread("./test_images/test6.jpg")
    # Run the function
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(dir_binary, cmap="gray")
    ax2.set_title("Thresholded Grad. Dir.", fontsize=24)
    plt.show(block=True)
