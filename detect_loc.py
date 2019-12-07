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
            'test': DataLoader(dataset['test'], batch_size=self.args.batch_size, shuffle=True)
        }
        
        os.makedirs(self.args.output_path, exist_ok=True)

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        num_anchors = cfg.LOC.NUM_ANCHORS
        self.yolo_layers = []
        for index in range(int(len(cfg.LOC.ANCHORS) / num_anchors)):
            self.yolo_layers.append(YoloLayer(cfg.LOC.ANCHORS[index*num_anchors: (index+1)*num_anchors],
                                              cfg.LOC.NUM_CLASSES,
                                              img_size=cfg.H))

    def inference(self):

        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()

        classes = ["D1", "D2", "D3", "D4", "D5", "S"]
        imgs, img_detections, img_names = [], [], []
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for batch_i, (x, x_name) in enumerate(self.dataloader['test']):
            imgs.append(x)
            img_names.append(x_name[0])
            x = x.float()
            x = x.to(self.device)

            with torch.no_grad():
                yolo = self.model(x)
                yolo_out = []

                for i in range(len(yolo)):
                    yolo_out.append(self.yolo_layers[i].forward(yolo[i]))
                yolo_out = torch.cat(yolo_out, 1)

                detection = non_max_suppression(yolo_out, 0.7, 0.4)
                img_detections.extend(detection)

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        for img_i, (img, img_name, img_detection) in enumerate(zip(imgs, img_names, img_detections)):
            print("(%d) Image Under Detection" % (img_i))

            plt.figure()
            plt.imshow(img.numpy()[0, 0], cmap='gray')

            if img_detection is not None:
                unique_labels = img_detection[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)

                """
                    START
                    Modify img_detection to ignore duplicated disks
                    Since each image could only contain one type of unique disks
                """
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
                """
                    END
                """

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in new_detections:

                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    plt.gca().add_patch(bbox)
                    plt.text(
                        (0.7 * x1),
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top"
                    )

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(f"output/{img_name}.jpg", bbox_inches="tight", pad_inches=0.0)
            plt.show()
            plt.close()


def main():
    # argparse
    parser = argparse.ArgumentParser(description="U-Net parameter selection")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=cfg.TRAIN.DEVICE)
    parser.add_argument("--output_path", type=str, default=cfg.TEST.OUTPUT_PATH)
    parser.add_argument("--model_path", type=str, default="./saves/loc_ckpt_2.pth")
    args = parser.parse_args()

    dataset_path = './datasets/localization/'
    testset = None, None
    dataset = {
        'test': testset
    }
    set_path = os.path.join(dataset_path, "test.txt")
    dataset["test"] = SpineLoadImageDataset(list_path=set_path)

    # model
    model = ResUnet(in_channels=cfg.IN_CH, out_channels=cfg.SEG.OUT_CH, init_features=cfg.INIT_FEATURES,
                    num_anchors=3, num_classes=6)
    # Training
    inference = Inference(args, dataset, model)
    inference.inference()


if __name__ == '__main__':
    main()

