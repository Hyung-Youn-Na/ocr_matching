"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np

import detector.scenetext_detection.craft.craft_utils as craft_utils
import detector.scenetext_detection.craft.imgproc as imgproc
import detector.scenetext_detection.craft.file_utils as file_utils
import json
import zipfile

from detector.scenetext_detection.craft.craft import CRAFT

from collections import OrderedDict

class CRAFT_Infer:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='CRAFT'):
        self.config = self._parse_config()
        self.results = dict()
        self.model_name = model
        self.model, self.refine_net = self._load_model()

    def _load_model(self):
        net = CRAFT()  # initialize

        print('Loading weights from checkpoint (' + self.config['trained_model'] + ')')
        if self.config['cuda']:
            net.load_state_dict(copyStateDict(torch.load(self.config['trained_model'])))
        else:
            net.load_state_dict(copyStateDict(torch.load(self.config['trained_model'], map_location='cpu')))

        if self.config['cuda']:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        refine_net = None
        if self.config['refine']:
            from detector.scenetext_detection.craft.refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.config['refiner_model'] + ')')
            if self.config['cuda']:
                refine_net.load_state_dict(copyStateDict(torch.load(self.config['refiner_model'])))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(self.config['refiner_model'], map_location='cpu')))

            refine_net.eval()
            self.config['poly'] = True

        return net, refine_net

    def inference_by_image(self, pil_image):
        t0 = time.time()

        # resize
        image = np.array(pil_image)

        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.config['canvas_size'],
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.config['mag_ratio'])
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.config['cuda']:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys, confidences = craft_utils.getDetBoxes(score_text, score_link, self.config['text_threshold'], self.config['link_threshold'], self.config['low_text'],
                                               self.config['poly'])

        if self.config['char_bbox']:
            char_boxes, char_polys, char_confidences = craft_utils.getDetBoxes(score_text, score_link, self.config['text_threshold'],
                                                                self.config['link_threshold'], self.config['low_text'],
                                                                self.config['poly'])

            char_boxes = craft_utils.adjustResultCoordinates(char_boxes, ratio_w, ratio_h)
            char_polys = craft_utils.adjustResultCoordinates(char_polys, ratio_w, ratio_h)

            for k in range(len(char_polys)):
                if char_polys[k] is None: char_polys[k] = char_boxes[k]

            char_xy_format_bboxes = []
            for bbox in boxes:
                xmin = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                xmax = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
                ymin = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                ymax = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
                char_xy_format_bboxes.append([xmin, ymin, xmax, ymax])

        else:
            char_xy_format_bboxes = None
            char_confidences = None

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        # render_img = score_text.copy()
        # render_img = np.hstack((render_img, score_link))
        # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.config['show_time']: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        xy_format_bboxes = []
        for bbox in boxes:

            xmin = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            xmax = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            ymin = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            ymax = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            xy_format_bboxes.append([xmin,ymin,xmax,ymax])

        return xy_format_bboxes, confidences, char_xy_format_bboxes, char_confidences


    def _parse_config(self):
        with open(os.path.join(self.path, 'config/config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        return config

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

if __name__ == '__main__':
    test = CRAFT_Infer()

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    img = Image.open('/nfs_shared/STR_Data/RoadView/sample3/파노라마_2/2019/19631104043_E_B.png').convert('RGB')
    bboxes, polys, score_text = test.inference_by_image(img)

    img = np.array(img)

    file_utils.saveResult('', img[:, :, ::-1], polys, dirname='./result/')