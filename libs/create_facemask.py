# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2022-06-25 22:23:11
    @Brief  : 生成戴口罩人脸数据集
"""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "libs")
from tqdm import tqdm
from facemask.wearmask import FaceMaskCreator
from pybaseutils import file_utils, image_utils


class FaceMaskDemo(object):
    def __init__(self):
        self.mask_creator = FaceMaskCreator(detect_face=True, alignment=False)

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
            mask, face_rects = self.mask_creator.create_masks(image, mask_type="random", vis=vis)
            if out_dir:
                self.mask_creator.save_image(image, mask, face_rects, out_dir, image_id)


if __name__ == '__main__':
    image_dir = "./facemask/test_image"  # 人脸图片
    out_dir = "./output"  # 生成戴口罩人脸输出目录
    fm = FaceMaskDemo()
    fm.create_wear_mask_faces(image_dir, out_dir, vis=True)
