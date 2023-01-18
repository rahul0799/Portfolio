"""This module is used to generate contours around information present on payslip.
"""
import statistics
import cv2
import numpy as np
from util import Util
from Config.config import ConfigParser
from PIL import Image, ImageEnhance
from catch_exceptions import *
from logger_contour import get_logger

cont_logger = get_logger("contour_module.detect_contour")


class DetectContour:
    """
    This class is used to detect coordinates of words using contours.
    """

    @staticmethod
    def validate_image_array(img):
        """
        Validates the img argument of EnsembleModel class

        :param img: the image_array argument
        :type img: `numpy.ndarray`

        :raises TypeError: When image_array is not of `numpy.ndarray` datatype

        :return: Array form of the image
        :rtype: `numpy.ndarray`
        """

        if isinstance(img, np.ndarray) is False:
            cont_logger.error("Input image is not in correct format")
            raise TypeError

        return img

    def __init__(self, img):
        """
        This constructor is called when the DetectContour class is initialised

        :param img: numpy array of the image whose contours have to be extracted
        :type img: `numpy.ndarray`

        :raises Exception: Exception in calling util class
        :raises Exception: Exception in resizing images
        :raises Exception: Exception in calling ConfigParser class
        """

        self.img = self.validate_image_array(img)
        try:
            self.ob1 = Util()
        except Exception as e:
            cont_logger.error("Error in calling util class")
            raise CallingUtilititesClass from e

        try:
            self.ob2 = ConfigParser()
        except Exception as e:
            cont_logger.error("Error in calling ConfigParser")
            raise CallingConfigParser from e

        try:
            self.img = self.ob1.resize_img(self.img, self.ob2.get_image_size())
        except Exception as e:
            cont_logger.error("Error in resizing image")
            raise ErrorResizing from e

        try:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ErrorColor from e

    def contours_to_bb(self, img):
        """
        This method is used to the contours in the image to bounding box.


        :param img: Image after removing noise.
        :type img: `numpy.ndarray`

        :return boundRect: Bounding Box coordinates.
        :rtype boundRect: `list`
        """

        canny_edged_img = self.ob1.canny_edge(img, 0, 255)

        contours = self.ob1.find_contour(canny_edged_img)[0]

        contours_from_left_to_right = sorted(
            contours, key=lambda x: x[0][0][0], reverse=False
        )

        contours_poly = [None] * len(contours_from_left_to_right)
        boundrect = [None] * len(contours_from_left_to_right)
        centers = [None] * len(contours_from_left_to_right)
        radius = [None] * len(contours_from_left_to_right)

        for idx, cnt in enumerate(contours_from_left_to_right):
            contours_poly[idx] = cv2.approxPolyDP(cnt, 1, True)
            boundrect[idx] = cv2.boundingRect(contours_poly[idx])
            centers[idx], radius[idx] = cv2.minEnclosingCircle(contours_poly[idx])

        return boundrect

    @staticmethod
    def remove_colour(img):
        """This method is used to remove the colour from the image.

        :param img: The input img param.
        :type img: `numpy.ndarray`

        :return img_np: Image after removing colour.
        :rtype : `numpy.ndarray`
        """
        try:
            im_pil = Image.fromarray(img)
        except Exception as e:
            cont_logger.error("Error in converting numpy.darray image to PIL image")
            raise ErrorConvertingToPIL from e
        converter = ImageEnhance.Color(im_pil)
        width, height = im_pil.size
        im_pil = converter.enhance(0)
        pixel_map = im_pil.load()
        for i in range(width):
            for idx_j in range(height):
                red, green, blue = im_pil.getpixel((i, idx_j))
                if (
                    red in range(160, 200)
                    and green in range(160, 200)
                    and blue in range(160, 200)
                ):
                    pixel_map[i, idx_j] = (255, 255, 255)
        try:
            img_np = np.asarray(im_pil)
        except Exception as e:
            cont_logger.error("Error in converting PIL image to numpy image")
            raise ErrorConvertingToNumpy from e
        return img_np

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        """This method is used to check if two boxes intersect.

        :param boxA: The input boxA param.
        :type boxA: `list`
        :param boxB: The input boxB param.
        :type boxB: `list`

        :return iou: Intersection over union area.
        :rtype : `float`
        """
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        interarea = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
        if interarea == 0:
            return 0
        boxaarea = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        boxbarea = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        iou = interarea / float(boxaarea + boxbarea - interarea)
        return iou

    def check_overlap(self, final_boxes):
        """
        This method is used to find the overlapping boxes and fix them.

        :param: input boxes param.
        :type: `list`

        :return: boxes returned after fixing overlap.
        :rtype: `list`
        """
        remo_overlap = []
        value = []
        added = []
        for i in range(len(final_boxes)):
            temp = final_boxes[i]
            for idx_j in range(len(final_boxes)):
                if i != idx_j:
                    temp2 = final_boxes[idx_j]
                    if (
                        self.bb_intersection_over_union(temp, temp2) > 0
                        and temp2 not in added
                        and temp not in added
                    ):
                        if temp not in remo_overlap:
                            remo_overlap.append(temp)
                        if temp2 not in remo_overlap:
                            remo_overlap.append(temp2)
                        point1 = min(temp[0], temp2[0])
                        point2 = min(temp[1], temp2[1])
                        point3 = max(temp[2], temp2[2])
                        point4 = max(temp[3], temp2[3])
                        value.append([point1, point2, point3, point4])

                        added.append(temp)
                        added.append(temp2)
        remo_box = []

        for i, temp in enumerate(remo_overlap):
            if temp not in remo_box:
                final_boxes.remove(temp)
                remo_box.append(temp)

        for i in value:
            final_boxes.append(i)

        return final_boxes, remo_overlap

    @staticmethod
    def merge_box(box):
        """
        This method is used to merge nearby boxes.

        :param: input box param.
        :type: `list`

        :return: final coordinates after merging
        :rtype: `list`
        """
        dic = {}
        ndic = []
        added = []
        flag = 0
        for i in range(len(box)):
            temp = box[i]
            near = []
            for idx_j in range(len(box)):
                if i != idx_j:
                    temp1 = box[idx_j]
                    if (
                        (
                            temp1[1]
                            in range(
                                temp[1] - (temp[3] - temp[1]),
                                temp[1] + (temp[3] - temp[1]),
                            )
                        )
                        and (
                            temp1[3]
                            in range(
                                temp[3] - (temp[3] - temp[1]),
                                temp[3] + (temp[3] - temp[1]),
                            )
                        )
                        and (0 <= temp1[0] - temp[2] < 8)
                    ):
                        near.append(idx_j)
                        added.append(idx_j)
                        break
            if near:
                dic[i] = near
                flag = 1
            else:
                if i not in added:
                    ndic.append(i)

        final_dic = {}
        checked = []
        count = 0

        for i in dic:
            if i not in checked:
                temp = dic[i][0]
                checked.append(i)
                final_check = []
                final_check.append(temp)
                while 1:
                    count = count + 1
                    if count > 1000:
                        raise InfiniteLoop
                    if temp in dic.keys():
                        checked.append(temp)
                        temp = dic[temp][0]
                        final_check.append(temp)
                    else:
                        break
                final_dic[i] = final_check

        final_boxes = []
        for i in final_dic:
            temp = final_dic[i]
            fbb = []

            point1 = box[i][0]
            point2 = box[i][1]
            point3 = box[i][2]
            point4 = box[i][3]
            for idx_j in temp:
                if point1 > box[idx_j][0]:
                    point1 = box[idx_j][0]
                if point2 > box[idx_j][1]:
                    point2 = box[idx_j][1]
                if point3 < box[idx_j][2]:
                    point3 = box[idx_j][2]
                if point4 < box[idx_j][3]:
                    point4 = box[idx_j][3]
            fbb.extend((point1, point2, point3, point4))
            final_boxes.append(fbb)

        for i in ndic:
            final_boxes.append(box[i])

        remo2 = []

        for i in range(len(final_boxes)):
            temp2 = final_boxes[i]
            for idx_j in range(len(final_boxes)):
                if i != idx_j:
                    temp3 = final_boxes[idx_j]
                    if (
                        temp3[0] >= temp2[0]
                        and temp3[1] >= temp2[1]
                        and temp3[2] <= temp2[2]
                        and temp3[3] <= temp2[3]
                    ):
                        if temp3 not in remo2:
                            remo2.append(temp3)

        for i in remo2:
            try:
                final_boxes.remove(i)
            except:
                raise ErrorRemovingBox

        return final_boxes, flag

    @staticmethod
    def split_exception_box(box_index, min_height, boxes):
        """
        This method is used to split exception boxes based on min_height
        received.

        :param box_index: index of the exception box.
        :type: `int`
        :param min_height: minimum height which should be to calculate ratio
                        to split the exception box.
        :type: `float`
        :param boxes: input param box in [xmin,ymin,xmax,ymax]
        :type: `list`

        :return: list of boxes that need to be added.
        :type: `list`
        """
        try:
            if (
                float((boxes[box_index][3] - boxes[box_index][1]) / min_height)
                - int((boxes[box_index][3] - boxes[box_index][1]) / min_height)
                >= 0.75
            ):
                ratio1 = round((boxes[box_index][3] - boxes[box_index][1]) / min_height)
            else:
                ratio1 = int((boxes[box_index][3] - boxes[box_index][1]) / min_height)
            add = []

            for j in range(ratio1):
                add.append(
                    [
                        int(boxes[box_index][0]),
                        int(
                            boxes[box_index][1]
                            + j * ((boxes[box_index][3] - boxes[box_index][1]) / ratio1)
                        ),
                        int(boxes[box_index][2]),
                        int(
                            boxes[box_index][3]
                            - (ratio1 - j - 1)
                            * (boxes[box_index][3] - boxes[box_index][1])
                            / ratio1
                        ),
                    ]
                )
        except:
            cont_logger.error("Error in spltting exception boxes")

        return add

    @staticmethod
    def find_exception_box(heights):
        """
        This method is used to find exception boxes.

        :param: input parma height is a dictionary with
                index as keys and height as values of a particular box
        :type: `dict`

        :return excep: fixed box coordinates in [xmin,xmax,ymin,ymax]
        :type: `list`
        :return min_height: minimum height by the excption boxes must be removed.
        :type: `float`
        """
        try:
            height_list = list(heights.values())

            if len(heights) % 2 != 0:
                min_height = statistics.median(height_list)
            else:
                height_list.sort()
                min_height = min(
                    height_list[int(len(height_list) / 2) - 1],
                    height_list[int(len(height_list) / 2)],
                )
            excep = []
            for i in heights:
                if float(heights[i] / min_height) >= 1.75:
                    excep.append(i)
        except:
            cont_logger.error("Error in finding exception boxes")

        return excep, min_height

    def region_wise_height(self, boxes):
        """
        This method is used to find exception boxes and fix them based on
        region wise average height of boxes.

        :param: input param box in [xmin,ymin,xmax,ymax]
        :type: `list`

        :return: fixed box coordinates in [xmin,ymin,xmax,ymax]
        :type: `list`
        """

        checked = []
        dic = {}
        for counti, i in enumerate(boxes):
            temp = []
            for countj, j in enumerate(boxes):
                if countj != counti and countj not in checked and counti not in checked:
                    if j[0] - i[0] < 600 and j[1] - i[1] < 100:
                        temp.append(countj)
                        checked.append(countj)
            checked.append(counti)
            if temp:
                dic[counti] = temp

        add = []
        remo_boxes = []
        for i in dic:
            heights = {}
            heights[i] = boxes[i][3] - boxes[i][1]
            if dic[i]:
                for value in dic[i]:
                    heights[value] = boxes[value][3] - boxes[value][1]
            if len(heights) > 1:
                excep, min_height = self.find_exception_box(heights)
            for box_index in excep:
                remo_boxes.append(boxes[box_index])
                add.extend(self.split_exception_box(box_index, min_height, boxes))

        for box in remo_boxes:
            try:
                boxes.remove(box)
            except:
                raise ErrorRemovingBox

        for box in add:
            boxes.append(box)

        return boxes

    def find_contours(self):
        """
        This method is used to find the contours in the image.

        :return: cnt_final contains the final contour image.
        :rtype: `numpy.ndarray`
        :return: final_boxes contains the final bounding ox coordinates.
        :rtype: `list`
        """
        img = self.img.copy()
        try:
            img = self.remove_colour(img)
            cont_logger.info("Successfully removed colour from image")
        except:
            cont_logger.error("Error in removing colour")
            raise ErrorRemovingColour

        try:
            imagedata_original = self.ob1.remove_lines(img)
            cont_logger.info("Successfully removed lines from image")
        except:
            cont_logger.error("Error in resizing image")

        try:
            grayscale_img = self.ob1.get_grayscale(imagedata_original)
            cont_logger.info("Succesfully converted image to grayscale")
        except:
            cont_logger.error("Error in converting image to grayscale")

        # try:
        #     gausBlur = self.ob1.add_gaussianblur(
        #         grayscale_img, self.ob2.get_gausblur_thres()
        #     )
        #     cont_logger.info("Succesfully applied gausian blur to image" + "\n\n")
        # except:
        #     cont_logger.error("Error in applying gausblur to image" + "\n\n")

        try:
            th2 = self.ob1.get_thres(grayscale_img)
            cont_logger.info("Succesfully applied threshold to image")
        except:
            cont_logger.error("Error in applying threshold to image")

        # try:
        #     erosion_op = self.ob1.remove_noise(th2)
        #     cont_logger.info("Succesfully removed noise from image" + "\n\n")
        # except:
        #     cont_logger.error("Error in removing threshold to image" + "\n\n")

        try:
            boundrect = self.contours_to_bb(th2)
            cont_logger.info(
                "Succesfully received bounding boxes corresponding to contours"
            )
        except:
            cont_logger.error("Error in getting bb from contours")

        box = []

        for i in range(len(boundrect)):
            points = []
            if (
                int(boundrect[i][3]) * int(boundrect[i][2]) > self.ob2.get_min_area_bb()
                and boundrect[i][3] / boundrect[i][2] < 8
            ):

                points.extend(
                    (
                        int(boundrect[i][0]),
                        int(boundrect[i][1]),
                        int(boundrect[i][0]) + boundrect[i][2],
                        int(boundrect[i][1]) + boundrect[i][3],
                    )
                )
                box.append(points)
        try:
            box = self.ob1.remove_box_inside_box(box)
            cont_logger.info("Succesfully removed boxes inside other boxes")
        except:
            cont_logger.error("Error in removing boxes inside other boxes")

        count = 0
        while 1:
            count = count + 1
            if count < 500:
                box, remo = self.check_overlap(box)
                if not remo:
                    break
            else:
                cont_logger.error("Encountered infinite loop while removing overlap")
                raise InfiniteLoop

        cont_logger.info("Succesfully fixed overlapping boxes")

        flag = 1

        final_boxes = box
        count = 0
        while flag:
            if count < 500:
                final_boxes, flag = self.merge_box(final_boxes)
                count = count + 1
            else:
                cont_logger.error("Encountered infinite loop while merging boxes")
                raise InfiniteLoop

        cont_logger.info("Succesfully merged nearby boxes")
        try:
            final_boxes = self.ob1.remove_exceptions(final_boxes)
            cont_logger.info("Succesfully removed exception boxes")
        except:
            cont_logger.error("Error in removing exception boxes")

        try:
            final_boxes = self.region_wise_height(final_boxes)
            cont_logger.info("Sucessfully Completed removing vertically merged box")
        except:
            cont_logger.error("Error in region_wise_height function")

        temp = []

        for box in final_boxes:
            point1 = box[0]
            point2 = box[2]
            point3 = box[1]
            point4 = box[3]
            temp.append([point1, point2, point3, point4])

        final_boxes = temp

        for i in range(len(final_boxes)):
            cv2.rectangle(
                self.img,
                (int(final_boxes[i][0]), int(final_boxes[i][2])),
                (int(final_boxes[i][1]), int(final_boxes[i][3])),
                (0, 0, 0),
                2,
            )

        return self.img, final_boxes
