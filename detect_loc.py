import os
import argparse
import time
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from cfgs.config import cfg
from models import *
from utils.datasets import *
from utils.loss import *
from utils.logger import *
from utils.utils import *
from evaluate import *

class Inference(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = {
            'test': DataLoader(dataset['test'], batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset['test'].collate_fn)
        }

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def inference(self):
        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()

        classes = ["D1", "D2", "D3", "D4", "D5", "S"]
        imgs, img_detections = [], []
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for batch_i, (x, _) in enumerate(self.dataloader['test']):
            imgs.append(x)
            x = x.float()
            x = x.to(self.device)

        '''
        ### TODO: BUG ISSUES NEED TO BE FIXED:
        ### REASONS: The output or yolo layers is xywh, but the required output for detection 
        ###          should be xyxy
        '''

            with torch.no_grad():
                yolo_list = self.model(x)
                yolo_output = []
                for i in range(3):
                    yolo_layer = YoloLayer(cfg.LOC.ANCHORS[3*i:3*(i+1)], cfg.LOC.NUM_CLASSES, img_size=512)
                    yolo_output.append(yolo_layer.forward(yolo_list[i]))
                yolo_output = (torch.cat(yolo_output, 1)).detach().cpu()

                detection = non_max_suppression(yolo_output, 0.1, 0.4)
                img_detections.extend(detection)

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        for img_i, (img, img_detection) in enumerate(zip(imgs, img_detections)):
            print("(%d) Image Under Detection" % (img_i))
            plt.imshow(img.numpy()[0, 0], cmap='gray')

            if img_detection is not None:
                # Rescale boxes to original image
                unique_labels = img_detection[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
    
                # TODO: Only modified on the lumbar Dataset
                #### Since each image could only contain one type of unique disks
                unique_dict = {}
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in img_detection:
                    cur = classes[int(cls_pred)]
                    if cur not in unique_dict:
                        unique_dict[cur] = (x1, y1, x2, y2, conf, cls_conf, cls_pred)
                    else:
                        if cls_conf > unique_dict[cur][5]:
                            unique_dict[cur] = (x1, y1, x2, y2, conf, cls_conf, cls_pred)
                new_detections = torch.Tensor([])
                for k, v in unique_dict.items():
                    add_tensor = torch.Tensor([i for i in v])
                    new_detections = torch.cat((new_detections, add_tensor), 0)
                new_detections = new_detections.view(-1, 7)
                #### END
    
                # TODO: Only modified on the lumbar Dataset
                #### Since each image could only contain one type of unique disks
                # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in new_detections:
                #### END
    
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    
                    box_w = x2 - x1
                    box_h = y2 - y1
    
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    # ax.add_patch(bbox)
                    plt.gca().add_patch(bbox)
                    # Add label
                    # 1. Add text
                    plt.text(
                        (0.7 * x1),
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top"
                        # bbox={"color": color, "pad": 0},
                    )
                    # 2. OR Add anotation
                    # plt.annotate(
                    #     classes[int(cls_pred)],
                    #     xy=(x1, y1),
                    #     xytext=((3.0*x1-x2)/2.0, y1),
                    #     color='w',
                    #     arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='white')
                    # )


def main():
    # argparse
    parser = argparse.ArgumentParser(description="U-Net parameter selection")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=cfg.TRAIN.DEVICE)
    parser.add_argument("--model_path", type=str, default="./saves/loc_ckpt_145.pth")
    args = parser.parse_args()

    dataset_path = './datasets/localization/'
    testset = None, None
    dataset = {
        'test': testset
    }
    set_path = os.path.join(dataset_path, "test.txt")
    dataset["test"] = SpineLocDataset(list_path=set_path)

    # model
    model = ResUnet(in_channels=cfg.IN_CH, out_channels=cfg.SEG.OUT_CH, init_features=cfg.INIT_FEATURES,
                    num_anchors=3, num_classes=6)
    # Training
    inference = Inference(args, dataset, model)
    inference.inference()


if __name__ == '__main__':
    main()

