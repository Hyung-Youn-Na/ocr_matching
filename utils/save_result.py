import os
import datetime
from PIL import Image, ImageDraw



def save_text_detection_result(result_dic, td_result_path):
    pred_image = Image.open(result_dic['pred_img_path']).convert('RGB')
    query_image = Image.open(result_dic['query_img_path']).convert('RGB')

    pred_image_basename = os.path.basename(result_dic['pred_img_path']).split('.')[0]
    query_image_basename = os.path.basename(result_dic['query_img_path']).split('.')[0]

    pred_draw = ImageDraw.Draw(pred_image)
    query_draw = ImageDraw.Draw(query_image)

    old_size = pred_image.size
    new_size = (10 + 2 * old_size[0],old_size[1])

    concat_image = Image.new("RGB", new_size)

    for q_td_bbox, p_td_bbox in zip(result_dic['query_img_text_bboxes'], result_dic['pred_img_text_bboxes']):
        query_draw.rectangle(((q_td_bbox[0], q_td_bbox[1]), (q_td_bbox[2], q_td_bbox[3])), outline='red', width=3)
        pred_draw.rectangle(((p_td_bbox[0], p_td_bbox[1]), (p_td_bbox[2], p_td_bbox[3])), outline='red', width=3)

    concat_image.paste(pred_image, (0,0))
    concat_image.paste(query_image, (10 + old_size[0], 0))

    concat_image.save(os.path.join(td_result_path, f'{query_image_basename}_{pred_image_basename}.jpg'))


def save_object_detection_result(result_dic, od_result_path):
    pred_image = Image.open(result_dic['pred_img_path']).convert('RGB')
    query_image = Image.open(result_dic['query_img_path']).convert('RGB')

    pred_image_basename = os.path.basename(result_dic['pred_img_path']).split('.')[0]
    query_image_basename = os.path.basename(result_dic['query_img_path']).split('.')[0]


    pred_draw = ImageDraw.Draw(pred_image)
    query_draw = ImageDraw.Draw(query_image)

    old_size = pred_image.size
    new_size = (10 + 2 * old_size[0], old_size[1])

    concat_image = Image.new("RGB", new_size)

    for q_od_bbox, p_od_bbox in zip(result_dic['query_img_object_bboxes'], result_dic['pred_img_object_bboxes']):
        query_draw.rectangle(((q_od_bbox[0], q_od_bbox[1]), (q_od_bbox[2], q_od_bbox[3])), outline='red', width=3)
        pred_draw.rectangle(((p_od_bbox[0], p_od_bbox[1]), (p_od_bbox[2], p_od_bbox[3])), outline='red', width=3)

    concat_image.paste(pred_image, (0, 0))
    concat_image.paste(query_image, (10 + old_size[0], 0))

    concat_image.save(os.path.join(od_result_path, f'{query_image_basename}_{pred_image_basename}.jpg'))
