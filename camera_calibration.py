import argparse
import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SHAPE_X = 1200
IMAGE_SHAPE_Y = 720
NUM_X_CORNERS = 9
NUM_Y_CORNERS = 6


def calibrate_camera(visualize=False, save_examples=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NUM_Y_CORNERS * NUM_X_CORNERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NUM_X_CORNERS, 0:NUM_Y_CORNERS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')
    print("Processing {} images".format(len(images)))
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (NUM_X_CORNERS, NUM_Y_CORNERS), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (NUM_X_CORNERS, NUM_Y_CORNERS), corners, ret)
            if save_examples:
                img = cv2.resize(img, (int(IMAGE_SHAPE_X / 3), int(IMAGE_SHAPE_Y / 3)), interpolation=cv2.INTER_AREA)
                cv2.imwrite("./output_images/chessboard{}.jpg".format(idx), img)
            if visualize:
                cv2.imshow("Chessboard Image", img)
                cv2.waitKey(0)
        else:
            print("Could not find chessboard corners for {}".format(fname))

    if visualize:
        cv2.destroyAllWindows()

    img_size = (IMAGE_SHAPE_X, IMAGE_SHAPE_Y)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist


def undistort_image(image, dict):
    mtx = dict["mtx"]
    dist = dict["dist"]
    image = cv2.undistort(image, mtx, dist, None, mtx)
    return image


def save_calibration_data(mtx, dist):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./calibration.p", "wb"))


def load_calibration_data():
    dict = pickle.load(open("./calibration.p", mode="rb"))
    # print("dict={}".format(dict))
    return dict


def show_undistort(image_file, visualize=False, save_examples=False):
    # Test undistortion on an image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dict = load_calibration_data()
    dst = undistort_image(img, dict)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=24)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_examples:
        save_file_name = "undistorted_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera calibration for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize data calibration")
    parser.add_argument("-save", action="store_true", help="Save calibration images")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    mtx, dist = calibrate_camera(visualize, save_examples)
    save_calibration_data(mtx, dist)
    show_undistort("./camera_cal/calibration2.jpg", visualize, save_examples)
    show_undistort("./test_images/signs_vehicles_xygrad.png", visualize, save_examples)
