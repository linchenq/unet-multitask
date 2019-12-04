import os
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from cfgs.config import cfg
from models import *
from utils.datasets import *
from utils.loss import *
from utils.logger import *
from utils.utils import *
from utils.metrics import *
from evaluate import *

class Trainer(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = {
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset['train'].collate_fn),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset['valid'].collate_fn)
        }

        # prerequisites
        os.makedirs(self.args.log_path, exist_ok=True)
        os.makedirs(self.args.save_path, exist_ok=True)

        # Log settings
        self.logger = Logger(self.args.log_path)

        # load pretrained weights
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))
            else:
                self.logger.log_summary(mode="WARNING", msg="Unknown pretrained weights or models")

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function
        num_anchors = cfg.LOC.NUM_ANCHORS
        self.loss = []
        for index in range(int(len(cfg.LOC.ANCHORS) / num_anchors)):
            self.loss.append(YoloLoss(cfg.LOC.ANCHORS[index*num_anchors: (index+1)*num_anchors],
                                      cfg.LOC.NUM_CLASSES,
                                      img_size=cfg.H))

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=self.args.lr_scheduler_step,
                                                          gamma=0.1)

    def train(self):
        num_epochs = self.args.epoch

        for epoch in tqdm(range(num_epochs)):
            since = time.time()

            self.run_single_step(epoch)
            self.exp_lr_scheduler.step()

            time_elapsed = time.time() - since
            print("Epoch {}: {:.0f}m {:.0f}s".format(epoch, time_elapsed // 60, time_elapsed % 60))
            self.logger.log_summary(mode="INFO", msg="Epoch {}: {:.0f}m {:.0f}s".format(epoch, time_elapsed // 60, time_elapsed % 60))

    def run_single_step(self, epoch):
        self.model.train()

        for batch_i, (x, targets) in enumerate(self.dataloader['train']):
            x, targets = x.float(), targets.float()
            x, targets = x.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            yolo_loss, losses = 0, []
            epoch_metrics = {
                    "grid_size": [],
                    "total_loss": [],
                    "x": [], "y": [], "w": [], "h": [],
                    "conf": [], "cls": [],
                    "cls_acc": [],
                    "recall50": [], "recall75": [], "precision": [],
                    "conf_obj": [], "conf_noobj": []
            }
            with torch.enable_grad():
                yolo = self.model(x)
                for i in range(len(yolo)):
                    i_loss = self.loss[i].forward(yolo[i], targets)
                    losses.append(i_loss)
                    yolo_loss += i_loss

                    # metrics logger
                    for k, v in self.loss[i].metrics.items():
                        epoch_metrics[k].append(v)

                yolo_loss.backward()
                self.optimizer.step()

            # metrics
            out_metric = yolo_metrics(epoch=epoch, phase="train", metrics=epoch_metrics)
            print(out_metric)
            self.logger.log_summary(mode="INFO", msg=out_metric)

            # tensorboard
            vis_step = len(self.dataloader['train']) * epoch + batch_i
            vis_metrics = [("train/total_loss", np.array(epoch_metrics["total_loss"]).sum()),
                           ("train/x_loss", np.array(epoch_metrics["x"]).sum()),
                           ("train/y_loss", np.array(epoch_metrics["y"]).sum()),
                           ("train/w_loss", np.array(epoch_metrics["w"]).sum()),
                           ("train/h_loss", np.array(epoch_metrics["h"]).sum())]
            self.logger.list_of_scalars_summary(vis_metrics, vis_step)

        if epoch % self.args.eval_interval == 0:
            # TODO : FINISH EVAL PART
            pass

        # if epoch % self.args.eval_interval == 0:
        #     valid_loss, valid_metrics, valid_samples = evaluate(self.model, self.dataloader, self.device)
            # print_metrics(valid_metrics, valid_samples, 'valid', epoch)
            # self.log_table = save_metrics(valid_metrics, valid_samples, 'valid', epoch, self.log_table)
            # self.logger.scalar_summary('loss/valid', np.mean(valid_loss), epoch)

        if epoch % self.args.save_interval == 0:
            torch.save(self.model.state_dict(), f"./saves/loc_ckpt_%d.pth" % epoch)


def main():
    # argparse
    parser = argparse.ArgumentParser(description="ResUNet")
    parser.add_argument("--batch_size", type=int, default=cfg.TRAIN.BATCH_SIZE)
    parser.add_argument("--epoch", type=int, default=cfg.TRAIN.EPOCH)
    parser.add_argument("--lr", type=float, default=cfg.TRAIN.LR)
    parser.add_argument("--lr_scheduler_step", default=cfg.TRAIN.LR_SCHEDULER)

    parser.add_argument("--log_path", type=str, default=cfg.TRAIN.LOG_PATH)
    parser.add_argument("--save_path", type=str, default=cfg.TRAIN.SAVE_PATH)
    parser.add_argument("--eval_interval", type=int, default=cfg.TRAIN.EVAL_INTERVAL)
    parser.add_argument("--save_interval", type=int, default=cfg.TRAIN.SAVE_INTERVAL)

    parser.add_argument("--pretrained_weights", type=str, default=None)

    parser.add_argument("--device", type=str, default=cfg.TRAIN.DEVICE)
    args = parser.parse_args()

    # dataset prepration
    dataset_path = './datasets/localization/'
    trainset, validset, testset = None, None, None
    dataset = {
        'train': trainset,
        'valid': validset,
    }
    for name in ['train', 'valid']:
        set_path = os.path.join(dataset_path, f"{name}.txt")
        dataset[name] = SpineLocDataset(list_path=set_path)

    # model
    model = ResUnet(in_channels=cfg.IN_CH,
                    out_channels=cfg.SEG.OUT_CH,
                    init_features=cfg.INIT_FEATURES,
                    num_anchors=cfg.LOC.NUM_ANCHORS,
                    num_classes=cfg.LOC.NUM_CLASSES)

    # Training
    trainer = Trainer(args, dataset, model)
    trainer.train()


if __name__ == '__main__':
    main()

