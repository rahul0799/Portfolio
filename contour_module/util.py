"""
This module helps provide utility functions for finding and optimizing bounding boxes.
"""
from math import ceil
import cv2
import numpy as np
from Config.config import ConfigParser
from catch_exceptions import CallingConfigParser, HeightError, ErrorRemovingBox
from logger_contour import get_logger

cont_logger = get_logger("contour_module.util")
# config = configparser.ConfigParser()
# config.read('./Config/config.ini')


class Util:
    """
    This class is used to extract the call some important functions needed in both modules.
    """

    def __init__(self):
        try:
            self.ob1 = ConfigParser()
        except:
            cont_logger.error("Error in calling config class.")
            raise CallingConfigParser

    @staticmethod
    def remove_noise(img):
        """
        Removes noise from image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after removing noise
        :rtype: `numpy.ndarray`
        """
        try:
            kernel = np.ones((1, 2), np.uint8)

            erosion_op = cv2.erode(img, kernel, iterations=1)

            # dilation_op = cv2.dilate(erosion_op, kernel, iterations=1)

            return erosion_op
        except:
            cont_logger.error("Error in calling config class.")
            raise TypeError

    @staticmethod
    def canny_edge(img, param1, param2):
        """
        Finds the edges in the image.

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after finding edges
        :rtype: `numpy.ndarray`
        """

        return cv2.Canny(img, param1, param2, apertureSize=5, L2gradient=True)

    @staticmethod
    def find_contour(img):
        """
        cv2 function to find the contours in the image.

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: list of contours found
        :rtype: `list`
        """

        return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    @staticmethod
    def resize_img(img, num):
        """
        Resizes the image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after resizing
        :rtype: `numpy.ndarray`
        """

        return cv2.resize(img, (num, num))

    @staticmethod
    def add_gaussianblur(img, kernel_size):
        """
        Adds Gaussian Blur to image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after adding Blur
        :rtype: `numpy.ndarray`
        """

        return cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

    @staticmethod
    def get_grayscale(img):
        """
        Converts image into grayscale

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after converting to grayscale
        :rtype: `numpy.ndarray`
        """

        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def get_thres(img):
        """
        Does thresholding on image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: image after appying thresholding
        :rtype: `numpy.ndarray`
        """

        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def remove_horizontal(self, img):
        """
        Removes horizontal lines from the image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: contours of the lines found
        :rtype: `list`
        """

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        detected_lines = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )
        cnts = self.find_contour(detected_lines)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        return cnts

    def remove_vertical(self, img):
        """
        Removes vertical lines from the image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :return: contours of the lines found
        :rtype: `list`
        """

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        detected_lines = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, vertical_kernel, iterations=2
        )
        cnts = self.find_contour(detected_lines)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        return cnts

    @staticmethod
    def draw_contour(cnts, img):
        """
        Removes vertical lines from the image

        :param img: the image_array argument
        :type img: `numpy.ndarray`
        :param cnts: contains contour information
        :return img: image after drawing white contour lines
        :rtype: `numpy.ndarray`
        """

        for cnt in cnts:
            cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)

        return img

    def remove_lines(self, img):
        """
        This method is used to remove the horizontal and vertical lines present in the img.

        :return img: image after removing vertical and horizontal lines
        :rtype img: `numpy.ndarray`
        """
        try:
            gray = self.get_grayscale(img)
            thresh = self.get_thres(gray)
            cnts_v = self.remove_vertical(thresh)
            img = self.draw_contour(cnts_v, img)
            cnts_h = self.remove_horizontal(thresh)
            img = self.draw_contour(cnts_h, img)
            return img
        except:
            cont_logger.error("Error in removing lines.")

    @staticmethod
    def remove_exceptions(final_boxes):
        """
        This method is used to optimize the solution by removing exceptions in bounding boxes.

        :param final_boxes: list of the bounding box coordinates that have to be worked on
        :type final_boxes: `list`
        :return final_boxes: contains the final bounding box coordinates.
        :rtype final_boxes: `list`
        """
        avg_height = 0

        for idx in range(len(final_boxes)):
            avg_height = avg_height + abs(
                int(final_boxes[idx][1]) - int(final_boxes[idx][3])
            )

        try:
            avg_height = avg_height / len(final_boxes)
        except:
            raise HeightError

        remo = []
        for idx in range(len(final_boxes)):
            temp = final_boxes[idx]
            height = int(abs(temp[1] - temp[3]))
            if height < ceil(avg_height / 1.5):
                remo.append(temp)

        for idx in remo:
            try:
                final_boxes.remove(idx)
            except:
                raise ErrorRemovingBox

        return final_boxes

    @staticmethod
    def remove_box_inside_box(box):
        """
        This method is used to optimize the solution by merging near by bounding
        boxes and rejecting bounding boxes based on area, if the area of the
        bounding box is too small then it is rejected.

        :param box: list of the bounding box coordinates that have to be extracted
        :type box: `list`
        :return final_boxes: contains the final bounding box coordinates.
        :rtype final_boxes: `list`
        """

        remo = []
        for idx in range(len(box)):
            temp = box[idx]
            for idx_j in range(len(box)):
                if idx != idx_j:
                    temp2 = box[idx_j]
                    if (
                        temp[0] >= temp2[0]
                        and temp[1] >= temp2[1]
                        and temp[2] <= temp2[2]
                        and temp[3] <= temp2[3]
                    ):
                        if temp not in remo:
                            remo.append(temp)

        for idx in remo:
            try:
                box.remove(idx)
            except:
                cont_logger.error("Error in remove box inside box logic.")
                raise ErrorRemovingBox

        return box

    @staticmethod
    def fix_height(box):
        """
        This method is used to optimize the solution by checking for merged boxes
        and trying to fix those cases by comparing the height with height of nearby boxes.

        :param box: list of the bounding box coordinates that have to be extracted
        :type box: `list`
        :return final_boxes: contains the final bounding box coordinates.
        :rtype final_boxes: `list`
        """

        dic = {}
        added = []

        for idx in range(len(box)):
            temp = box[idx]
            near = []
            for idx_j in range(len(box)):
                if idx != idx_j and idx_j not in added:
                    temp1 = box[idx_j]
                    if (
                        abs(temp1[1] - temp[1]) < 5
                        and abs(temp1[0] - temp[0]) < 20
                        and int(abs(temp[0] - temp[2]) * abs(temp[1] - temp[3])) > 20
                        and int(abs(temp1[0] - temp1[2]) * abs(temp1[1] - temp1[3]))
                        > 20
                    ):
                        near.append(idx_j)
                        added.append(idx_j)
                    break
            if near:
                dic[idx] = near
            added.append(idx)

        heights = []
        final_dic = {}
        checked = []
        count = 0
        for idx in dic:
            if idx not in checked:
                temp = dic[idx][0]
                checked.append(idx)
                final_check = []
                final_check.append(temp)
                while 1:
                    count = count + 1
                    if temp in dic.keys() and count < 1000:
                        checked.append(temp)
                        temp = dic[temp][0]
                        final_check.append(temp)
                    else:
                        cont_logger.error(
                            "Error in fix height logic while grouping boxes."
                        )
                        break
                final_dic[idx] = final_check

        mean_heights = []
        remo = []
        for idx in final_dic:
            # if int(i) == 5:
            temp = final_dic[idx]
            heights = {}
            min_height = int(box[idx][3] - box[idx][1])
            heights[idx] = int(box[idx][3] - box[idx][1])
            mean = 0
            flag = 0
            excep_count = 0
            for idx_j in temp:
                if min_height > int(box[idx_j][3] - box[idx_j][1]):
                    min_height = int(box[idx_j][3] - box[idx_j][1])
                heights[idx_j] = int(box[idx_j][3] - box[idx_j][1])
            for idx_z in heights:
                if heights[idx_z] > min_height * 2:
                    remo.append(idx_z)
                    flag = 1
                    excep_count = excep_count + 1
                else:
                    mean = mean + (heights[idx_z])
            if flag == 1:
                counter = excep_count
                while counter:
                    mean_heights.append(mean / (len(heights) - excep_count))
                    counter -= 1

        add = []
        for count, idx in enumerate(remo):
            temp = box[idx]
            height = int(temp[3] - temp[1])
            ratio = int(abs(temp[1] - temp[3]) / (mean_heights[count]))
            for idx_j in range(ratio):
                add.append(
                    [
                        int(temp[0]),
                        int(temp[1] + idx_j * height / ratio),
                        int(temp[2]),
                        int(temp[3] - (ratio - idx_j - 1) * height / ratio),
                    ]
                )

        remo_box = []
        for idx in remo:
            remo_box.append(box[idx])

        for idx in remo_box:
            try:
                box.remove(idx)
            except:
                cont_logger.error("Error in removing boxes.")
                raise ErrorRemovingBox

        for idx in add:
            box.append(idx)

        return box
