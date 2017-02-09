import argparse
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration
import rgb_to_hls
import sobel
import transform


# Edit this function to create your own pipeline.
def pipeline(img):
    img = np.copy(img)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, dict)

    # Perspective transform
    img, M, Minv = transform.apply_transform(img)

    # HLS S-channel threshold binary
    hls_binary = rgb_to_hls.hls_select(img, thresh=(190, 250))

    # Sobel threshold binary
    sobel_binary = sobel.combined_sobel(img)

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might be beneficial to replace this channel with something else.
    # color_binary = np.dstack((np.zeros_like(sobel_binary), sobel_binary, hls_binary))
    color_binary = cv2.bitwise_or(hls_binary, sobel_binary)
    return color_binary


# Edit this function to create your own pipeline.
def pipeline2(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Undistort image
    dict = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, dict)

    # Perspective transform
    img, M, Minv = transform.apply_transform(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary


def show_pipeline(image_file, visualize=False, save_example=False):
    # Read in an image
    image = mpimg.imread(image_file)
    # Run the function
    # pipeline_image = pipeline2(image, s_thresh=(190, 250), sx_thresh=(40, 100))
    pipeline_image = pipeline(image)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=24)
    ax2.imshow(pipeline_image, cmap="gray")
    ax2.set_title("Pipeline Result", fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_example:
        save_file_name = "pipeline_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Complete image pipeline for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize pipeline image")
    parser.add_argument("-save", action="store_true", help="Save pipeline image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    # show_pipeline("./test_images/test1.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test2.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test3.jpg", visualize, save_examples)
    show_pipeline("./test_images/test4.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test5.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test6.jpg", visualize, save_examples)
