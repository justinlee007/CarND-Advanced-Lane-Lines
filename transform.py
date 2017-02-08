# M = cv2.getPerspectiveTransform(src, dst)
# Minv = cv2.getPerspectiveTransform(dst, src)
# warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

import cv2
import matplotlib.pyplot as plt
import numpy as np

import camera_calibration


def apply_transform(img):
    img_shape = (img.shape[1], img.shape[0])
    src, dst = create_warp_mappings(img)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_shape)
    return warped, M


# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, dict):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found:
    # a) draw corners
    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    # Note: you could pick any four of the detected corners
    # as long as those four corners define a rectangle
    # One especially smart way to do this would be to use four well-chosen
    # corners that were automatically detected during the undistortion steps
    # We recommend using the automatic detection of corners in your code
    # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    # e) use cv2.warpPerspective() to warp your image to a top-down view

    # Use the OpenCV undistort() function to remove distortion

    undist = camera_calibration.undistort_image(img, dict)

    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    warped = None
    M = None
    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def create_warp_mappings(image):
    shape = image.shape
    vertical_offset = 90
    horizontal_offset = 40
    mid_vertical = shape[0] / 2
    mid_horizontal = shape[1] / 2
    bottom_left = [horizontal_offset, (shape[0] - horizontal_offset)]
    middle_left = [(mid_horizontal - (horizontal_offset * 3)), (mid_vertical + vertical_offset)]
    middle_right = [(mid_horizontal + (horizontal_offset * 3)), (mid_vertical + vertical_offset)]
    bottom_right = [(shape[1] - horizontal_offset), (shape[0] - horizontal_offset)]
    src = np.float32([middle_right, bottom_right, bottom_left, middle_left])
    dst = np.float32([[shape[1], 0], [shape[1], shape[0]], [0, shape[0]], [0, 0]])
    return src, dst


if __name__ == '__main__':
    # Read in an image
    img = cv2.imread('./test_images/test2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dict = camera_calibration.load_calibration_data()

    img = camera_calibration.undistort_image(img, dict)

    src, dst = create_warp_mappings(img)
    warped, M = apply_transform(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    for p in src:
        ax1.plot(p[0], p[1], 'ro')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped)
    for p in dst:
        ax2.plot(p[0], p[1], 'go')
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)
