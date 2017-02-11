import argparse
import os

import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import camera_calibration
import imaging
import transform
from LineTracker import LineTracker
from Lanes import Lanes
import poly_fit

line_tracker = LineTracker(center_dis=(0.25 * 1280), window_width=25, window_height=40, padding=25, slide_res=5,
                           capture_height=720)

lanes = Lanes(debug_mode=False)


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

    # result = process_image(img, processed_image, Minv)

    # lanes.locate_lanes(processed_image)

    # Fit polynomials and set lane x/y arrays
    # lanes.fit_lanes()

    # Check curvature sanity

    # Draw lines back onto road
    # result = lanes.draw_lanes(img, processed_image, Minv)

    return result


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    if ((center - width / 2) < img_ref.shape[1]) & ((center + width / 2) > 0):
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width)):min(img_ref.shape[1], int(center + width))] = 1
    return output


def process_image(img, warped, Minv):
    img_size = (img.shape[1], img.shape[0])
    # for debugging just to get the binary image
    # warped = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    # return warped

    # find the best line centers based on the binary pixel of interest input
    frame_centers = line_tracker.track_line(warped)
    # need these parameters to draw the graphic overlay illustraing the window convolution matching
    window_width = line_tracker.window_width
    window_height = line_tracker.window_height
    # points used for graphic overlay
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # points used to find the left and right lanes
    rightx = []
    leftx = []

    res_yvals = np.arange(warped.shape[0] - (window_height + window_height / 2), 0, -window_height)

    for level in range(1, len(frame_centers)):
        l_mask = window_mask(window_width, window_height, warped, frame_centers[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, frame_centers[level][1], level)
        # add center value found in frame to the list of lane points per left,right
        leftx.append(frame_centers[level][0])
        rightx.append(frame_centers[level][1])
        # fill in graphic points here if pixels fit inside the specificed window from l/r mask
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # drawing the graphic overlay to represents the results found for tracking window centers
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, zero_channel, template)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    graphic_measure = cv2.addWeighted(warpage, 0.2, template, 0.75, 0.0)

    # fit the lane boundaries to the left,right center positions found
    yvals = range(0, warped.shape[0])

    left_fit = np.polyfit(res_yvals, leftx, 3)
    left_fitx = left_fit[0] * yvals * yvals * yvals + left_fit[1] * yvals * yvals + left_fit[2] * yvals + left_fit[3]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 3)
    right_fitx = right_fit[0] * yvals * yvals * yvals + right_fit[1] * yvals * yvals + right_fit[2] * yvals + right_fit[
        3]
    right_fitx = np.array(right_fitx, np.int32)

    # used to find center curve
    curve_xpts = [(right_fitx[0] + left_fitx[0]) / 2,
                  (right_fitx[len(right_fitx) / 2] + left_fitx[len(left_fitx) / 2]) / 2,
                  (right_fitx[-1] + left_fitx[-1]) / 2]
    curve_ypts = [yvals[0], yvals[(int)(len(yvals) / 2)], yvals[-1]]
    curve_fit = np.polyfit(curve_ypts, curve_xpts, 2)
    curve_fitx = curve_fit[0] * yvals * yvals + curve_fit[1] * yvals + curve_fit[2]

    # used to format everything so its ready for cv2 draw functions
    left_lane = np.array(list(
        zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(
        zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(
        zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    curve_pts = np.array([list(zip(curve_fitx, yvals))], np.int32)

    # draw lane lines, middle curve, road background on two different blank overlays
    road = np.zeros_like(template)
    road_bkg = np.zeros_like(template)
    cv2.fillPoly(road, [left_lane], color=[9, 67, 109])
    cv2.fillPoly(road, [right_lane], color=[9, 67, 109])
    cv2.polylines(road, [curve_pts], isClosed=False, color=[5, 176, 249], thickness=3)
    for horz_line_y in line_tracker.horz_lines:
        cv2.line(road, (left_fitx[(int)(horz_line_y)], (int)(horz_line_y)),
                 (right_fitx[(int)(horz_line_y)], (int)(horz_line_y)), color=[5, 176, 249], thickness=3)

    cv2.fillPoly(road_bkg, [inner_lane], color=[38, 133, 197])

    # after done drawing all the marking effects, warp back image to its orginal perspective.
    # Note for the two different overlays, just seperating road_warped and road_warped_bkg to get two different alpha values, its just for astetics...
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    # merging all the different overlays, basically make things look pretty!
    lane_template = np.array(cv2.merge((road_warped[:, :, 2], road_warped[:, :, 2], road_warped[:, :, 2])), np.uint8)
    bkg_template = np.array(cv2.merge((road_warped_bkg[:, :, 2], road_warped_bkg[:, :, 2], road_warped_bkg[:, :, 2])),
                            np.uint8)
    base = cv2.addWeighted(img, 1.0, bkg_template, -0.6, 0.0)
    base = cv2.addWeighted(base, 1.0, road_warped_bkg, 0.6, 0.0)
    base = cv2.addWeighted(base, 1.0, lane_template, -1.8, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.9, 0.0)
    # return result

    # calcuate the middle line curvature
    ym_per_pix = line_tracker.ym_per_pix  # meters per pixel in y dimension
    xm_per_pix = line_tracker.xm_per_pix  # meteres per pixel in x dimension
    curve_fit_cr = np.polyfit(np.array(curve_ypts, np.float32) * ym_per_pix,
                              np.array(curve_xpts, np.float32) * xm_per_pix, 2)
    curverad = ((1 + (2 * curve_fit_cr[0] * curve_ypts[1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])
    line_tracker.curvatures.append(curverad)
    curverad = line_tracker.smooth_curve()

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # add text backdrop
    txt_bkg = np.zeros_like(result)
    txt_bkg_pts = np.array(
        [(0, 0), ((int)(result.shape[1] * .5), 0), ((int)(result.shape[1] * .5), (int)(result.shape[0] * .25)),
         (0, (int)(result.shape[0] * .25))])
    cv2.fillPoly(txt_bkg, [txt_bkg_pts], color=[200, 200, 200])
    result = cv2.addWeighted(result, 1.0, txt_bkg, -1.0, 0.0)
    # draw the text showing curvature, offset, and speed
    cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (5, 176, 249), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (5, 176, 249), 2)

    # scale the graphic measure that we generated near the start for finding window centers, by some constant factor in both axis
    g_scale = 0.4
    graphic_overlay = cv2.resize(graphic_measure, (0, 0), fx=g_scale, fy=g_scale)
    g_xoffset = result.shape[1] - graphic_overlay.shape[1]
    # overlay the graphic measure in the result image at the top right corner
    result[:graphic_overlay.shape[0], g_xoffset:g_xoffset + graphic_overlay.shape[1]] = graphic_overlay

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
    show_pipeline("./test_images/test1.jpg", visualize, save_examples)
    show_pipeline("./test_images/test2.jpg", visualize, save_examples)
    show_pipeline("./test_images/test3.jpg", visualize, save_examples)
    show_pipeline("./test_images/test4.jpg", visualize, save_examples)
    show_pipeline("./test_images/test5.jpg", visualize, save_examples)
    show_pipeline("./test_images/test6.jpg", visualize, save_examples)
    # process_video()
