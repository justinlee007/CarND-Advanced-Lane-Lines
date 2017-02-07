import argparse
import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

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
                cv2.imwrite("./output_images/chessboard{}.jpg".format(idx), img)
            if visualize:
                cv2.imshow("Chessboard Image", img)
                cv2.waitKey(0)
        else:
            print("Could not find chessboard corners for {}".format(fname))

    cv2.destroyAllWindows()

    # Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration2.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=24)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=24)
    if visualize:
        plt.show(block=True)
    if save_examples:
        f.savefig("./output_images/example_undist.png", bbox_inches="tight")
    return mtx, dist


def save_calibration_data(mtx, dist):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./calibration.p", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera calibration for Udacity Advanced Lane Finding project")
    parser.add_argument("-show", action="store_true", help="Visualize data calibration")
    parser.add_argument("-save", action="store_true", help="Save calibration images")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)
    mtx, dist = calibrate_camera(visualize, save_examples)
    save_calibration_data(mtx, dist)
