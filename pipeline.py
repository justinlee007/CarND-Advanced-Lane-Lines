import argparse
import os

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import cv2
import glob
import camera_calibration
import imaging
import transform
from Lanes import Lanes
import poly_fit

lanes = Lanes(debug_mode=False)


def pipeline(img):
    """
    Complete pipeline for lane finding.
    The image passed in is first undistorted.  Then the image is transformed or warped.
    Next the image passes through sobel and color filtering.  After that, a histogram finds the left and right lane.
    Next, the lanes are painted and the image warping is reversed.
    Finally, text is overlayed to display radius of curvature and distance from center.
    :param img:
    :return:
    """
    # Undistort image
    dict = camera_calibration.load_calibration_data()
    img = camera_calibration.undistort_image(img, dict)

    # Perspective transform
    image, M, Minv = transform.apply_transform(img)

    # Sobel/color thresholding and gaussian blur
    processed_image = imaging.process_imaging(image)

    lanes.locate_lanes(processed_image)
    lanes.fit_lanes()
    # result = lanes.draw_lanes(img, processed_image, Minv)

    # Find lanes
    histogram_image, left_fit, right_fit = poly_fit.histogram(processed_image)
    lane_lines, left_line_pts, right_line_pts = poly_fit.lane_lines(processed_image, left_fit, right_fit)

    # Undo perspective transform
    inv_warp = cv2.warpPerspective(lane_lines, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(img, 1, inv_warp, 0.4, 0)

    # Overlay radius of curvature and distance from center
    result = lanes.draw_text(result)

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

    process_video()
    # show_pipeline("./test_images/test1.jpg", visualize, save_examples)
    images = glob.glob("./test_images/test*.jpg")
    for file_name in images:
        show_pipeline(file_name, visualize, save_examples)
