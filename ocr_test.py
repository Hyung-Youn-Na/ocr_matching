import os
import random
import datetime
from collections import OrderedDict

import pprint
import numpy

from PIL import Image, ImageDraw, ImageFont
from detector.scenetext_detection.craft.CRAFT_Infer import CRAFT_Infer
from recognizer.scenetext_recognizer.deeptext.DeepText import DeepText

from utils import Logging, merge_filter, save_result


class EdgeModule:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.area_threshold = 0.5
        self.result = dict()
        self.td_model, self.tr_model = self.load_models()

    def _parse_settings(self, dict_settings):
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return td_option, tr_option

    def load_models(self):
        td_model = CRAFT_Infer()
        tr_model = DeepText()

        return td_model, tr_model

    def inference_by_image_recognition_before(self, image):
        result = {
            'image_path': image
        }

        img = Image.open(image)

        print('TD model inference...')
        img_text_bboxes, img_td_confidences, _, _ = self.td_model.inference_by_image(img)

        result['img_text_bboxes'] = img_text_bboxes

        img_group_texts = []
        img_group_text_confidences = []
        print('TR model inference...')
        for text_bbox in img_text_bboxes:
            crop_img = img.crop(tuple(text_bbox))
            texts, tr_confidences, _ = self.tr_model.inference_by_image([crop_img])
            img_group_texts.extend(texts)
            img_group_text_confidences.extend(tr_confidences)

        result['img_group_texts'] = img_group_texts
        result['img_group_text_confidences'] = img_group_text_confidences

        self.result = result


    def plot_result(self, final_result_path):
        image = Image.open(self.result['image_path']).convert('RGB')

        image_basename = os.path.basename(self.result['image_path']).split('.')[0]

        image_size = image.size
        new_size = (image_size[0], 800 + image_size[1])
        image_border = Image.new("RGB", new_size)


        image_draw = ImageDraw.Draw(image)

        fontpath = "/nfs_shared/STR_Data/graduate_project/utils/Gulim.ttf"
        font = ImageFont.truetype(fontpath, 50)
        font_small = ImageFont.truetype(fontpath, 30)

        text = []

        for text_bbox, text in \
            zip(self.result['img_text_bboxes'], self.result['img_group_texts']):
            print(text)
            image_draw.rectangle(((text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])), outline='red', width=3)
            text_position = (text_bbox[0], max(0, text_bbox[1]-30))
            text_left, text_top, text_right, text_bottom = image_draw.textbbox(text_position, text, font=font_small)
            image_draw.rectangle((text_left - 5, text_top - 5, text_right + 5, text_bottom + 5), fill="red")
            image_draw.text(text_position, text, font=font_small, fill="black")

        image.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))

        # save_result.save_text_detection_result(self.result, td_result_path)
        #
        # save_result.save_object_detection_result(self.result, od_result_path)


        # pred_border.save('pred_image.jpg')
        # query_border.save('query_image.jpg')




if __name__ == '__main__':
    # result_path = '/nfs_shared/STR_Data/graduate_project/results_detector/'
    result_path = './'

    td_result_path = '/hdd/ocr_matching/result/'
    od_result_path = '/nfs_shared/STR_Data/graduate_project/ObjectDetection/result/'
    nowDate = datetime.datetime.now()
    nowDate_str = nowDate.strftime("%Y-%m-%d-%H-%M-%S")
    nowDate_result_path = os.path.join(result_path, nowDate_str)

    if not os.path.exists(nowDate_result_path):
        os.mkdir(nowDate_result_path)

    td_result_path = os.path.join(td_result_path, nowDate_str)

    if not os.path.exists(td_result_path):
        os.mkdir(td_result_path)

    od_result_path = os.path.join(od_result_path, nowDate_str)

    if not os.path.exists(od_result_path):
        os.mkdir(od_result_path)

    main = EdgeModule()
    q = '/hdd/ocr_matching/cubemap_sample/'
    q_paths = os.listdir(q)
    for path in q_paths:
        print(path)
        if '.png' not in path and '.jpg' not in path:
            continue
        main.inference_by_image_recognition_before(os.path.join(q, path))
        main.plot_result(td_result_path)
        # pprint.pprint(main.result)
