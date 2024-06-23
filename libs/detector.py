# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-08-22 16:19:37
# --------------------------------------------------------
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import cv2
import numpy as np
from pybaseutils import image_utils, file_utils, debug

project_root = os.path.join(os.path.dirname(__file__))


class Detector(object):

    def __init__(self, detect_type="face"):
        self.class_names = None
        self.detect_type = detect_type
        if self.detect_type == "face":
            sys.path.append(os.path.join(os.path.dirname(__file__), "light_detector"))
            from light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "face"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "rfb"
            input_size = [320, 320]
            priors_type = "face"
            device = args.device
            model_path = "light_detector/data/pretrained/pth/rfb-face-mask.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=device)
        elif self.detect_type == "face640":
            sys.path.append(os.path.join(os.path.dirname(__file__), "light_detector"))
            from light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "face"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "rfb"
            input_size = [640, 640]
            priors_type = "face"
            device = args.device
            model_path = "light_detector/data/pretrained/pth/rfb_face_640_640.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=device)
        elif self.detect_type == "person":
            sys.path.append(os.path.join(os.path.dirname(__file__), "light_detector"))
            from light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "person"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "rfb"
            input_size = [640, 360]
            priors_type = "person"
            device = args.device
            model_path = "light_detector/data/pretrained/pth/rfb_person_640_360.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=device)

        elif self.detect_type == "face_person":
            sys.path.append(os.path.join(os.path.dirname(__file__), "light_detector"))
            from light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "face", "person"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "mbv2"
            input_size = [640, 360]
            priors_type = "face_person"
            device = args.device
            model_path = "light_detector/data/pretrained/pth/mbv2_face_person_640_360.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=device)
        else:
            raise Exception("Error:{}".format(self.detect_type))
        print("detect_type:{},class_names:{}".format(self.detect_type, self.class_names))

    def start_capture(self, video_path, save_video=None, detect_freq=1, isshow=True):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
        video_cap = image_utils.get_video_capture(video_path)
        width, height, numFrames, fps = image_utils.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_utils.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.task(frame, isshow)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def task(self, frame, isshow=True):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox_score = self.detect(rgb_image, isshow=False)
        if isshow:
            image, boxes, probs = self.show_image(rgb_image, bbox_score)

    def detect(self, rgb_image, isshow=False):
        """
        :param rgb_image:
        :param isshow:
        :return:
        """
        bbox_score = np.asarray([])
        boxes, labels, probs = self.detector.detect_image(rgb_image, isshow=False)
        if len(boxes) > 0:
            bbox_score = np.hstack((boxes, probs.reshape(-1, 1)))
        if isshow:
            self.show_image(rgb_image, bbox_score)
        return bbox_score, labels

    def detect_image_dir(self, image_dir, isshow=True):
        """
        :param image_dir: directory or image file path
        :param isshow:<bool>
        :return:
        """
        if os.path.isdir(image_dir):
            image_list = file_utils.get_files_lists(image_dir, postfix=["*.jpg", "*.png"])
        elif os.path.isfile(image_dir):
            image_list = [image_dir]
        else:
            raise Exception("Error:{}".format(image_dir))
        for img_path in image_list:
            orig_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            bbox_score, labels = self.detect(rgb_image, isshow=False)

            if isshow:
                image, boxes, probs = self.show_image(rgb_image, bbox_score, labels)
                self.save_result(img_path, image, boxes, probs)

    def show_image(self, image, bbox_score, labels, waitKey=0):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.class_names:
            labels = [self.class_names[l] for l in labels]
        if len(bbox_score) > 0:
            boxes = bbox_score[:, :4]
            probs = bbox_score[:, 4]
            image = image_utils.draw_image_bboxes_text(image, boxes, labels, color=(255, 0, 0))
        else:
            boxes, probs = np.asarray([]), np.asarray([])
        cv2.imshow("Det", image)
        cv2.waitKey(waitKey)
        return image, boxes, probs

    def save_result(self, img_path, image, boxes, probs):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "det-result")
        basename = os.path.basename(img_path).split(".")[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        out_json_path = os.path.join(out_dir, basename + ".json")
        json_data = {"boxes": boxes}
        file_utils.write_json_path(out_json_path, json_data)


if __name__ == "__main__":
    image_dir = "light_detector/data/test_images"
    det = Detector(detect_type="face")
    # det = Detector(detect_type="darknet")
    det.detect_image_dir(image_dir, isshow=True)
    # det.start_capture(video_path, isshow=True)
