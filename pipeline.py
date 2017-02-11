import argparse
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

import camera_calibration
import imaging
import poly_fit
import transform


# Edit this function to create your own pipeline.
def pipeline(img):
    img_size = (img.shape[1], img.shape[0])
    # Undistort image
    dict = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(img)

    # Run the function
    grad_binary_x = imaging.abs_sobel_thresh(image, orient='x', thresh=(40, 255))
    grad_binary_y = imaging.abs_sobel_thresh(image, orient='y', thresh=(25, 255))

    grad_binary = cv2.bitwise_and(grad_binary_x, grad_binary_y)

    color_binary = imaging.color_threshold(image, hls_thresh=(150, 255), hsv_thresh=(200, 255))

    processed_image = cv2.bitwise_or(grad_binary, color_binary)

    processed_image = imaging.gaussian_blur(processed_image, kernel=9)

    poly_fit_result = poly_fit.poly_fit(processed_image)

    inv_warp = cv2.warpPerspective(poly_fit_result, Minv, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1, inv_warp, 0.5, 0)
    return result


def process_video(input_file="./project_video.mp4", output_file="./project_video_output.mp4"):
    clip = VideoFileClip(input_file)
    video_clip = clip.fl_image(pipeline)
    video_clip.write_videofile(output_file, audio=False)


def show_pipeline(image_file, visualize=False, save_example=False):
    # Read in an image
    image = mpimg.imread(image_file)
    # Run the function
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
    # show_pipeline("./test_images/test4.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test5.jpg", visualize, save_examples)
    # show_pipeline("./test_images/test6.jpg", visualize, save_examples)
    process_video()
