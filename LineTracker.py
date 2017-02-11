import numpy as np


class LineTracker:
    """
    Advance Lane Line Tracker used to generate input from camera images for autonomous cars
    Copied from Aaron Brown's Tracker: https://github.com/awbrown90/Advance-Lane-Finding/blob/master/LineTracker.py

    The curve template used per vertical level
    (left_window + right_window)

    (padding)                                                                (padding)           /\
     <------ |---window_width--|           (+)         |---window_width---|  -------->           |
             |---left_window---|_______________________|---right_window---|                      | level
           <------>  |----------------center_dis-----------------|   <------>                    |
         (slide_res)                                                (slide_res)                  -
    """

    # when starting a new instance please be sure to specify all unassigned variables
    def __init__(self, center_dis, window_width, window_height, padding, slide_res,
                 line_momentum=.5, height_res=[1, 2, 4], smooth_factor=15, capture_height=100, ym=1, xm=1,
                 line_dist=200):
        # list that stores all the past (left,right) center set values used for smoothing the output
        self.recent_centers = []

        # base center pixel distance to use, tracker uses this as the sort of expected mean for center distance
        self.center_dis = center_dis

        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = window_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = window_height

        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.padding = padding

        # how much to shrink / extend the center_dis base in pixels
        self.slide_res = slide_res

        # (0-1) percentage of how much line is allowed to slow down from moving in curving direction
        # examples:
        # if 0, the line could move vast distance in curving direction and then suddenly move exactly vertical but this is not a good natural curve shape
        # if 1, the line curving progression is constant or increasing, line is not allowed to decrease speed, this can easily throw it off path though
        # A good value for momentum is something in the middle of 0-1
        self.line_momentum = line_momentum

        # the image is broken into levels from dividing image input height by window height.
        # centers on bottom level are always calculated by level between 1-(top level) can be searched using different paths by different level skipping increments
        # This approach just helps to ensure the most curves are considered so better chance to return the max value curve
        # 1 is most standard but it could lead down a wrong path from noisy pixels so with 2,4 level increments as well there is a better chance to get the optimal curve
        # Note inbetween levels from height_res > 1 are interpolated linearly.
        self.height_res = height_res

        # How many previous best curve sets to average over to get smooth results.
        self.smooth_factor = smooth_factor

        # The following parameters are used for the speed tracking algorithm that uses template matching to measure
        # changes in distance traveled per frame

        # image height of the reference to calculate speed and do horizontal line overlay
        self.capture_height = capture_height

        self.ym_per_pix = ym  # meters per pixel in vertical axis

        self.xm_per_pix = xm  # meters per pixel in horizontal axis

        self.line_dist = (int)(
            line_dist / self.ym_per_pix)  # how far apart horizontal lines are spaced apart in meters

        self.horz_lines = []  # list of horizontal lines to show distance tracking

        # keeps track of all previous radius measurments and then can smooth it
        self.curvatures = []

    # use averaging to return a more stable curvature value
    def smooth_curve(self):
        return np.average(self.curvatures[-self.smooth_factor:], axis=0)

    # the main tracking function for finding and storing lane segment positions
    def new_track(self, warped):

        collection_curve_centers = []
        collection_curve_max = []

        # Search reference sides
        for line_ref in [-1, 1]:  # [-1,1]: # -1 for left reference and 1 for right reference

            # Search curve directions
            for curve_direction in [-1, 1]:  # [-1,1]: # -1 for curving left and 1 for curving right

                # Search levels
                for height_res in self.height_res:  # iteration of different height resoultions where 1 is highest resoultion but might get side tracked

                    # local center storing to later find best picks
                    curve_centers = []
                    curve_max = 0

                    # if we dont have any reference positions to start we will just look at the highest amount of pixels in bottom quarter of image to get started
                    if (len(self.recent_centers) == 0):
                        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
                        l_center = int(np.argmax(l_sum))
                        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
                        r_center = int(np.argmax(r_sum)) + int(warped.shape[1] / 2)

                    # use previous start positions to make the lanetracker move more stable
                    else:
                        r_center = self.recent_centers[-1][0][1]
                        l_center = self.recent_centers[-1][0][0]

                        # sliding around previous start point to see if we can get a section with high pixels
                        l_sum = np.sum(
                            warped[int(17 * warped.shape[0] / 18):, l_center - self.padding:l_center + self.padding],
                            axis=0)
                        # kind of important: switch the ordering so that a higher basis is given to orginal position instead of the very left range
                        # otherwise this could cause sliding problems acutally when all pixels are very high
                        l_sum = l_sum * np.concatenate(
                            (np.linspace(.5, 1, self.padding), np.linspace(1, .5, self.padding)), axis=0)

                        # as before we needed to be careful when all pixels were very high, also need to do the same when pixels are very low
                        # basically if all range pixels are very high or low we prefer to use the old position and not try to slide...
                        if (len(l_sum) > 0) & (np.max(l_sum) > 100):
                            l_center = int(np.argmax(l_sum)) + (l_center - self.padding)

                        # same thing we did for the left start but now for the right start
                        r_sum = np.sum(
                            warped[int(17 * warped.shape[0] / 18):, r_center - self.padding:r_center + self.padding],
                            axis=0)
                        r_sum = r_sum * np.concatenate(
                            (np.linspace(.5, 1, self.padding), np.linspace(1, .5, self.padding)), axis=0)
                        if (len(r_sum) > 0) & (np.max(r_sum) > 100):
                            r_center = int(np.argmax(r_sum)) + (r_center - self.padding)

                    # used to control the lane curvature, helps create nice shaped splines
                    l_dis = 0
                    r_dis = 0

                    # add what we found for the first layer
                    curve_centers.append((l_center, r_center))

                    # go through each layer locking for max pixel locations
                    for level in range(1, (int)(warped.shape[0] / self.window_height), height_res):

                        # local center storing to later find best picks
                        max_values = []
                        max_pos = []

                        # used for sliding around centers, and we use the last layer centers as reference
                        # This is a pretty complicated routine, but basically going through coupled convolutions and taking notice if we are curving right/left, which left/right lane is the position references
                        # Main setup is to find local optimum lane positions and add them to a list and compare all of them for the highest value at the end.
                        for pad in range(-self.padding * 2, self.padding * 2, self.slide_res):
                            conv_template = np.concatenate((np.ones(self.window_width),
                                                            np.zeros(self.center_dis - self.window_width + pad),
                                                            np.ones(self.window_width)))
                            conv_signal = np.convolve(conv_template, np.sum(
                                warped[int(warped.shape[0] - (level + 1) * 40):int(warped.shape[0] - level * 40), :],
                                axis=0))
                            if ((((line_ref == 1) & (curve_direction == 1)) & (len(conv_signal[max(
                                            r_center + self.window_width / 2 + r_dis, 0):min(
                                            r_center + self.window_width / 2 + self.padding + r_dis,
                                warped.shape[1])]) != 0)) | (
                                        ((line_ref == 1) & (curve_direction == -1)) & (len(conv_signal[max(
                                                    r_center + self.window_width / 2 - self.padding + r_dis, 0):min(
                                                r_center + self.window_width / 2 + r_dis, warped.shape[1])]) != 0)) \
                                        | (((line_ref == -1) & (curve_direction == 1)) & ((len(conv_signal[max(
                                                    l_center + self.window_width / 2 + self.center_dis + pad + l_dis,
                                    0):min(
                                                        l_center + self.window_width / 2 + self.center_dis + pad + self.padding + l_dis,
                                    warped.shape[1])])) != 0)) | (
                                        ((line_ref == -1) & (curve_direction == -1)) & ((len(conv_signal[max(
                                                            l_center + self.window_width / 2 - self.padding + self.center_dis + pad + l_dis,
                                        0):min(
                                                        l_center + self.window_width / 2 + self.center_dis + pad + l_dis,
                                        warped.shape[1])])) != 0))):
                                if line_ref == 1:
                                    if curve_direction == 1:
                                        pos = np.argmax(conv_signal[
                                                        max(r_center + self.window_width / 2 + r_dis, 0):min(
                                                            r_center + self.window_width / 2 + self.padding + r_dis,
                                                            warped.shape[1])]) + max(r_center + r_dis, 0)
                                    else:
                                        pos = np.argmax(conv_signal[
                                                        max(r_center + self.window_width / 2 - self.padding + r_dis,
                                                            0):min(r_center + self.window_width / 2 + r_dis,
                                                                   warped.shape[1])]) + max(
                                            r_center - self.padding + r_dis, 0)
                                else:
                                    if curve_direction == 1:
                                        pos = np.argmax(conv_signal[max(
                                            l_center + self.window_width / 2 + self.center_dis + pad + l_dis, 0):min(
                                            l_center + self.window_width / 2 + self.center_dis + pad + self.padding + l_dis,
                                            warped.shape[1])]) + max(l_center + self.center_dis + pad + l_dis, 0)
                                    else:
                                        pos = np.argmax(conv_signal[max(
                                            l_center + self.window_width / 2 - self.padding + self.center_dis + pad + l_dis,
                                            0):min(l_center + self.window_width / 2 + self.center_dis + pad + l_dis,
                                                   warped.shape[1])]) + max(
                                            l_center - self.padding + self.center_dis + pad + l_dis, 0)
                                max_pos.append(pos)
                                if line_ref == 1:
                                    if curve_direction == 1:
                                        if abs(l_center - (pos - (self.center_dis + pad))) < (
                                                    self.padding + abs(r_dis)):
                                            max_value = np.amax(conv_signal[
                                                                max(r_center + self.window_width / 2 + r_dis, 0):min(
                                                                    r_center + self.window_width / 2 + self.padding + r_dis,
                                                                    warped.shape[1])])
                                        else:
                                            max_value = -10
                                    else:
                                        if abs(l_center - (pos - (self.center_dis + pad))) < (
                                                    self.padding + abs(+r_dis)):
                                            max_value = np.amax(conv_signal[max(
                                                r_center + self.window_width / 2 - self.padding + r_dis, 0):min(
                                                r_center + self.window_width / 2 + r_dis, warped.shape[1])])
                                        else:
                                            max_value = -10
                                else:
                                    if curve_direction == 1:
                                        if abs(r_center - pos) < (self.padding + abs(l_dis)):
                                            max_value = np.amax(conv_signal[max(
                                                l_center + self.window_width / 2 + self.center_dis + pad + l_dis,
                                                0):min(
                                                l_center + self.window_width / 2 + self.center_dis + pad + self.padding + l_dis,
                                                warped.shape[1])])
                                        else:
                                            max_value = -10
                                    else:
                                        if abs(r_center - pos) < (self.padding + abs(l_dis)):
                                            max_value = np.amax(conv_signal[max(
                                                l_center + self.window_width / 2 - self.padding + self.center_dis + pad + l_dis,
                                                0):min(l_center + self.window_width / 2 + self.center_dis + pad + l_dis,
                                                       warped.shape[1])])
                                        else:
                                            max_value = -10
                                max_values.append(max_value)
                            else:
                                if curve_direction == 1:
                                    max_pos.append(r_center + r_dis)
                                    max_values.append(10)
                                else:
                                    max_pos.append(l_center + (self.center_dis + pad) + l_dis)
                                    max_values.append(10)

                        pad = np.arange(-self.padding * 2, self.padding * 2, self.slide_res)
                        max_index = np.argmax(max_values)
                        r_dis = (max_pos[max_index] - r_center) * .5
                        l_dis = ((max_pos[max_index] - (self.center_dis + pad[max_index])) - l_center) * .5
                        r_center = max_pos[max_index]
                        l_center = r_center - (self.center_dis + pad[max_index])

                        # find centers and values of interpolated levels if height_res was more than 1
                        if (height_res > 1) & (level > 1):
                            right_level_xoffset = curve_centers[level - height_res][1] - r_center
                            left_level_xoffset = curve_centers[level - height_res][0] - l_center
                            for interpolated_level in range(level - height_res, level - 1):
                                interpolated_ratio = (interpolated_level - (level - height_res) + 1) / (height_res)
                                inter_r_center = r_center + right_level_xoffset * interpolated_ratio
                                inter_l_center = l_center + left_level_xoffset * interpolated_ratio

                                conv_template = np.ones(self.window_width)
                                conv_signal = np.convolve(conv_template, np.sum(warped[int(
                                    warped.shape[0] - (interpolated_level + 1) * 40):int(
                                    warped.shape[0] - interpolated_level * 40), :], axis=0))
                                right_interpolated_value = conv_signal[inter_r_center + self.window_width / 2]
                                left_interpolated_value = conv_signal[inter_l_center + self.window_width / 2]

                                curve_centers.append((inter_l_center, inter_r_center))
                                curve_max += (left_interpolated_value + right_interpolated_value)

                        curve_centers.append((l_center, r_center))
                        curve_max += np.amax(max_values)

                    collection_curve_centers.append(curve_centers)
                    collection_curve_max.append(curve_max)

        # return the highest valued centers positions out of all the local best candidates considered
        return collection_curve_centers[np.argmax(collection_curve_max)]

    def track_line(self, warped):

        self.recent_centers.append(self.new_track(warped))
        # return averaged values of the line centers, helps to keep the markers from jumping around too much
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
