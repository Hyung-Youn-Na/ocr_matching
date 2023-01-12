import string
import yaml
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from PIL import Image

from recognizer.scenetext_recognizer.deeptext.utils import CTCLabelConverter, AttnLabelConverter
from recognizer.scenetext_recognizer.deeptext.dataset import ListDataset, AlignCollate
from recognizer.scenetext_recognizer.deeptext.model import Model


class semi_opt(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class DeepText:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='CRAFT'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._parse_config()

        with open(self.config['character'], 'r') as charset:
            line = charset.readline()



        if self.config['sensitive']:
            with open(self.config['character'], 'r') as cf:
                cf_str = cf.read().rstrip()
                self.config['character'] = cf_str
        else:
            self.config['character'] = line + '0123456789abcdefghijklmnopqrstuvwxyz'

        self.results = dict()
        self.model_name = model
        self.model, self.converter = self._load_model()






    def _load_model(self):
        """ model configuration """
        if 'CTC' in self.config['Prediction']:
            converter = CTCLabelConverter(self.config['character'])
        else:
            converter = AttnLabelConverter(self.config['character'])
        self.config['num_class'] = len(converter.character)

        if self.config['rgb']:
            self.config['input_channel'] = 3

        opt = semi_opt(self.config)

        model = Model(opt)
        # print('model input parameters', self.config)
        model = torch.nn.DataParallel(model).to(self.device)

        # load model
        print('loading scene text recognizer pretrained model from %s' % self.config['saved_model'])
        model.load_state_dict(torch.load(self.config['saved_model'], map_location=self.device))

        model.eval()

        return model, converter


    def inference_by_image(self, pil_image_list):

        pred_texts = []

        AlignCollate_demo = AlignCollate(imgH=self.config['imgH'], imgW=self.config['imgW'],
                                         keep_ratio_with_pad=self.config['PAD'])
        demo_data = ListDataset(pil_image_list=pil_image_list, config=self.config)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=int(self.config['workers']),
            collate_fn=AlignCollate_demo, pin_memory=True)
        tr_confidences = []
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.config['batch_max_length']] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.config['batch_max_length'] + 1).fill_(0).to(self.device)

                if 'CTC' in self.config['Prediction']:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)

                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds_top5_str = []
                    preds = self.model(image, text_for_pred, is_train=False)
                    # print(preds.shape)
                    _, preds_top10 = preds.topk(10, 2)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)
                    for i in range(5):
                        preds_top5 = preds_top10[:, :,i]
                        preds_top_i = self.converter.decode(preds_top5, length_for_pred)
                        preds_top5_str.append(preds_top_i[0])
                    # print(preds_str, preds_top5_str)

                dashed_line = '-' * 80
                # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

                # print(f'{dashed_line}\n{head}\n{dashed_line}')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in self.config['Prediction']:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    if len(pred_max_prob.cumprod(dim=0)) == 0:
                        pred = ''
                        confidence_score = 0.0
                    else:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    pred_texts.append(pred)
                    tr_confidences.append(confidence_score)
                    # print(f'{img_name}\t{pred:25s}\t{confidence_score:0.4f}')
            return pred_texts, tr_confidences, preds_top5_str



    def _parse_config(self):
        with open(os.path.join(self.path, 'config/config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        return config



if __name__ == '__main__':

    test = DeepText()
    pil_image_list = []
    path_list = os.listdir('/workspace/recognizer/scenetext_recognizer/deeptext/demo_image/')
    for path in path_list:
        pil_image_list.append(Image.open('/workspace/recognizer/scenetext_recognizer/deeptext/demo_image/'+ path).convert('RGB'))
    print(len(pil_image_list))
    test.inference_by_image(pil_image_list=pil_image_list)
