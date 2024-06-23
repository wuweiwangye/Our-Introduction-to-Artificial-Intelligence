import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../../libs")
sys.path.insert(0, "libs")
import cv2
import math
import dlib
import numpy as np
import random
import face_recognition
from pybaseutils import file_utils, image_utils
# from detector import Detector
from libs.detector import Detector
from tqdm import tqdm
from PIL import Image, ImageFile

project_root = os.path.dirname(os.path.abspath(__file__))


class FaceMaskCreator(object):
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, template_dir=None, model='cnn', detect_face=True, alignment=False):
        """
        :param template_dir: 口罩模板
        :param model: choices=['hog', 'cnn'], Which face detection model to use.
        :param detect_face: 是否检测人脸
        :param alignment: 是否对人脸进行矫正
        """
        random.seed(200)
        template_dir = template_dir if template_dir else os.path.join(project_root, "template")
        self.model = model
        self.detect_face = detect_face
        self.alignment = alignment
        self.mask_templates = self.gat_mask_templates(template_dir)
        self.mask_types = list(self.mask_templates.keys())
        self.num_mask = len(self.mask_templates)
        print("have mask template:{}\n{}".format(self.num_mask, self.mask_types))
        self.predictor = dlib.shape_predictor(os.path.join(project_root, "dat/shape_predictor_68_face_landmarks.dat"))
        self.detector = Detector(detect_type="face")

    def gat_mask_templates(self, mask_dir):
        """
        prefix="blue-mask"
        :param mask_dir:
        :return:
        """
        mask_paths = file_utils.get_files_lists(mask_dir, postfix=["*.png"])
        mask_layers_list = {}
        for path in mask_paths:
            basename = os.path.basename(path)
            mask_layers_list[basename] = Image.open(path)
        return mask_layers_list

    def face_alignment(self, faces, out_size=None):
        """
        :param faces:
        :param out_size:
        :return:
        """
        # 预测关键点
        faces_aligned = []
        for face in faces:
            rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
            shape = self.predictor(np.uint8(face), rec)
            # left eye, right eye, nose, left mouth, right mouth
            order = [36, 45, 30, 48, 54]
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
            # 计算两眼的中心坐标
            eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,
                          (shape.part(36).y + shape.part(45).y) * 1. / 2)
            dx = (shape.part(45).x - shape.part(36).x)
            dy = (shape.part(45).y - shape.part(36).y)
            # 计算角度
            angle = math.atan2(dy, dx) * 180. / math.pi
            # 计算仿射矩阵
            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
            # 进行仿射变换，即旋转
            RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
            if out_size:
                RotImg = cv2.resize(RotImg, tuple(out_size), interpolation=cv2.INTER_AREA)
            faces_aligned.append(RotImg)
        return faces_aligned

    def detect_face_landmarks(self, image):
        """
        face_locations= (top, right, bottom, left)
        # (left,top,right,bottom)
        x = rect[3]
        y = rect[0]
        w = rect[1] - x
        h = rect[2] - y
        :param image:
        :return:
        """
        h, w, d = image.shape
        face_rects = []
        if self.detect_face:
            # (top(ymin), right(xmax), bottom(ymax), left(xmin))
            # face_rects = face_recognition.face_locations(image, model=self.model)
            dets, labels = self.detector.detect(image)  # dets is bbox_score
            dets = dets.astype(np.int)
            face_rects = [[ymin, xmax, ymax, xmin] for (xmin, ymin, xmax, ymax, label) in dets]
        face_rects = face_rects if len(face_rects) > 0 else [(0, w, h, 0)]
        face_lmaks = face_recognition.face_landmarks(image, face_rects)
        for i, rect in enumerate(face_rects):  # (ymin, xmax, ymax,xmin)（x,y,w,h）
            face_rects[i] = (rect[3], rect[0], rect[1] - rect[3], rect[2] - rect[0])
        return face_rects, face_lmaks

    def create_masks(self, image: np.ndarray, mask_type="random", vis=True):
        """
        :param image:
        :param mask_type:  mask name or random
        :param vis:
        :return:
        """
        face_rects, face_lmaks = self.detect_face_landmarks(image)
        mask_template = self.get_target_template(mask_type)  # mask face
        mask = image.copy()
        for lmak in face_lmaks:
            # image = self.add_masks_faces(image, lmak, mask_image=mask_template)
            mask = self.add_masks_faces_v2(mask, lmak, mask_image=mask_template)
        if vis:
            mask_faces = self.crop_masks_faces(mask, face_rects, alignment=self.alignment)
            self.show_image(image, mask, face_rects, face_lmaks, mask_faces)
        return mask, face_rects

    def get_target_template(self, mask_type):
        """
        :param mask_type:  mask name or random,train,val
        :return:
        """
        if mask_type == "random":
            mask_id = int(random.uniform(0, self.num_mask))
            mask_name = self.mask_types[mask_id]
            mask_image = self.mask_templates[mask_name]
        elif mask_type == "train":
            # create mask:1:1
            unmask_rate = 1
            mask_id = int(random.uniform(0, self.num_mask * (unmask_rate + 1)))
            # print(mask_id)
            if mask_id < self.num_mask:
                mask_name = self.mask_types[mask_id]
                mask_image = self.mask_templates[mask_name]
            else:
                mask_image = None
        elif mask_type == "val":
            mask_id = int(random.uniform(0, self.num_mask))
            mask_name = self.mask_types[mask_id]
            mask_image = self.mask_templates[mask_name]
        elif mask_type in self.mask_types:
            mask_image = self.mask_templates[mask_type]
        else:
            raise Exception("no mask:{}".format(mask_type))
        return mask_image

    def crop_masks_faces(self, image, face_rects, alignment=True, face_size=[112, 112]):
        face_rects = image_utils.extend_xywh(np.array(face_rects), scale=(1.2, 1.2))
        crop_faces = image_utils.get_rects_image(np.array(image), rects_list=face_rects, size=(None, None))
        # 人脸对齐操作并保存
        if alignment:
            crop_faces = self.face_alignment(crop_faces, face_size)
        return crop_faces

    def split_mask(self, mask_image, chin_left_point, chin_right_point, nose_point, chin_bottom_point, new_height):
        # left
        height, width, d = mask_image.shape
        width_ratio = 1.2
        mask_left_img = image_utils.get_bbox_crop_padding(mask_image, (0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = max(1, int(mask_left_width * width_ratio))
        mask_left_img = image_utils.resize_image(mask_left_img, size=(mask_left_width, new_height))

        # right
        mask_right_img = image_utils.get_bbox_crop_padding(np.asarray(mask_image), (width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = max(1, int(mask_right_width * width_ratio))
        mask_right_img = image_utils.resize_image(mask_right_img, size=(mask_right_width, new_height))
        return mask_left_img, mask_right_img, mask_left_width, mask_right_width

    def add_masks_faces(self, image, face_landmark: dict, mask_image: ImageFile):
        if mask_image is None:
            return image
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]
        # split mask and resize
        mask_image = np.asarray(mask_image)
        # new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
        new_height = max(1, int(np.linalg.norm(nose_v - chin_bottom_v)))
        mask_left_img, mask_right_img, mask_left_width, mask_right_width = self.split_mask(mask_image,
                                                                                           chin_left_point,
                                                                                           chin_right_point,
                                                                                           nose_point,
                                                                                           chin_bottom_point,
                                                                                           new_height)
        # merge mask
        shape = (new_height, mask_left_width + mask_right_width, 4)
        mask_img = np.zeros(shape=shape, dtype=np.uint8)
        mask_img = image_utils.cv_paste_image(mask_img, mask_left_img, (0, 0))
        mask_img = image_utils.cv_paste_image(mask_img, mask_right_img, (mask_left_width, 0))

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = image_utils.image_rotation(mask_img, angle)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = (mask_left_width + mask_right_width) // 2 - mask_left_width
        radian = angle * np.pi / 180
        box_x = max(0, center_x + int(offset * np.cos(radian)) - rotated_mask_img.shape[1] // 2)
        box_y = max(0, center_y + int(offset * np.sin(radian)) - rotated_mask_img.shape[0] // 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image = image_utils.cv_paste_image(image, mask_img, (box_x, box_y))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def add_masks_faces_v2(self, img, face_landmark: dict, mask_image: ImageFile):
        if mask_image is None:
            return img
        image = Image.fromarray(img)
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = mask_image.width
        height = mask_image.height
        width_ratio = 1.2
        new_height = max(1, int(np.linalg.norm(nose_v - chin_bottom_v)))

        # left
        mask_left_img = mask_image.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = max(1, int(mask_left_width * width_ratio))
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = mask_image.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = max(1, int(mask_right_width * width_ratio))
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        angle = 90 - angle * 180 / np.pi
        rotated_mask_img = mask_img.rotate(angle, expand=True)
        mask_img = rotated_mask_img
        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2
        # add mask
        image.paste(mask_img, (box_x, box_y), mask_img)
        image = np.asarray(image)
        return image

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)

    @staticmethod
    def get_face_lmaks_list(face_lmaks):
        out_face_lmaks = []
        for lmaks in face_lmaks:
            ls = []
            for l in lmaks.values():
                ls += l
            out_face_lmaks.append(ls)
        return out_face_lmaks

    def show_image(self, image, mask, face_rects, face_lmaks, mask_faces, delay=0):
        face_lmaks_ = self.get_face_lmaks_list(face_lmaks)
        image = image_utils.draw_landmark(image, face_lmaks_)
        image = image_utils.draw_image_rects(image, face_rects, color=(0, 0, 255))
        mask = image_utils.draw_image_rects(mask, face_rects, color=(255, 0, 0))
        vis = image_utils.image_hstack([image, mask])
        image_utils.cv_show_image("image-mask", vis, delay=1)
        for face in mask_faces:
            image_utils.cv_show_image("face-mask", face, delay=0)
        return vis

    def save_image(self, image, mask, face_rects, out_dir, image_id):
        """
        :param image:
        :param mask:
        :param face_rects:
        :param out_dir:
        :param image_id:
        :return:
        """
        image_faces = self.crop_masks_faces(image, face_rects, alignment=self.alignment)
        mask_faces = self.crop_masks_faces(mask, face_rects, alignment=self.alignment)
        out_file = file_utils.create_dir(out_dir, "image-mask", "{}.jpg".format(image_id))
        image_utils.save_image(out_file, mask)
        for i in range(len(face_rects)):
            mask_file = file_utils.create_dir(out_dir, "mask", "{}_{:0=3d}.jpg".format(image_id, i))
            image_file = file_utils.create_dir(out_dir, "nomask", "{}_{:0=3d}.jpg".format(image_id, i))
            image_utils.save_image(mask_file, mask_faces[i])
            image_utils.save_image(image_file, image_faces[i])

    def create_wear_mask_faces(self, image_dir, out_dir=None, vis=True):
        """
        生成戴口罩人脸数据集
        :param image_dir: 人脸图片目录
        :param out_dir:  生成戴口罩人脸输出目录
        :param vis: 是否可视化效果
        :return:
        """
        image_list = file_utils.get_files_lists(image_dir)
        for image_path in tqdm(image_list):
            image_id = os.path.basename(image_path).split(".")[0]
            image = image_utils.read_image(image_path, size=(512, None), use_rgb=True)
            mask, face_rects = self.create_masks(image, mask_type="random", vis=vis)
            if out_dir:
                self.save_image(image, mask, face_rects, out_dir, image_id)


if __name__ == '__main__':
    mask_dir = os.path.join(project_root, "template")  # 口罩模板
    image_dir = "./test_image"  # 人脸图片
    out_dir = "./output"  # 生成戴口罩人脸输出目录
    fm = FaceMaskCreator(mask_dir, alignment=False)
    fm.create_wear_mask_faces(image_dir, out_dir, vis=True)
