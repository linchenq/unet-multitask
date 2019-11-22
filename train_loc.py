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

        # load pretrained weights
        if self.args.pretrained_weights is not None:
            if self.args.pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.args.pretrained_weights))

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function
        num_anchors = 3
        self.loss = []
        for index in range(len(cfg.LOC.ANCHORS) / num_anchors):
            self.loss.append(YoloLoss(cfg.LOC.ANCHORS,
                                      cfg.LOC.NUM_CLASSES,
                                      img_size=__C.H))

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=self.args.lr_scheduler_step,
                                                          gamma=0.1)

        # Log settings
        self.logger = Logger(self.args.log_path)
        self.log_table = None

    def train(self):
        num_epochs = self.args.epoch
        loc_metrics = [
                "grid_size"
                "loss",
                "x",
                "y",
                "w",
                "h",
                "conf",
                "cls",
                "cls_acc",
                "recall50",
                "recall75",
                "precision",
                "conf_obj",
                "conf_noobj"
        ]

        for epoch in tqdm(range(num_epochs)):
            since = time.time()

            self.run_single_step(epoch, loc_metrics)
            self.exp_lr_scheduler.step()

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def run_single_step(self, epoch, loc_metrics):
        self.model.train()

        for batch_i, (x, targets) in enumerate(self.dataloader['train']):
            x, targets = x.float(), targets.float()
            x, targets = x.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            yolo_loss, losses = 0, []
            with torch.enable_grad():
                yolo = self.model(x)
                for i in range(len(yolo)):
                    i_loss = self.loss[i].forward(yolo[i], targets)
                    losses.append(i_loss)
                    yolo_loss += i_loss

                yolo_loss.backward()
                self.optimizer.step()


            # log_str = "---- [Phase %s][Epoch %d/%d, Batch %d/%d] ----"% \
            #             ('train', epoch, self.args.epoch, batch_i, len(self.dataloader['train']))
            # print(log_str)
            total_loss = 0
            losses = []

            x, targets = x.float(), targets.float()
            x, targets = x.to(self.device), targets.to(self.device)
            # x, targets = Variable(x.to(self.device)), Variable(targets.to(self.device), requires_grad=False)

            self.optimizer.zero_grad()

            # forward & backward
            with torch.enable_grad():
                yolo_list = self.model(x)

                for i in range(3):
                    _loss = self.loss[i].forward(yolo_list[i], targets)
                    losses.append(_loss)

                total_loss = sum(losses)
                total_loss.backward()
                self.optimizer.step()

                print(f"---- [Total loss: {total_loss.detach().cpu().item()}]")


        # print_metrics(metrics, samples, 'train', epoch)
        # self.log_table = save_metrics(metrics, samples, 'train', epoch, self.log_table)
        # epoch_loss = metrics['loss'] / samples

        # self.logger.scalar_summary('loss/train', np.mean(loss_train), epoch)

        # if epoch % self.args.eval_interval == 0:
        #     valid_loss, valid_metrics, valid_samples = evaluate(self.model, self.dataloader, self.device)
            # print_metrics(valid_metrics, valid_samples, 'valid', epoch)
            # self.log_table = save_metrics(valid_metrics, valid_samples, 'valid', epoch, self.log_table)
            # self.logger.scalar_summary('loss/valid', np.mean(valid_loss), epoch)
        if epoch % self.args.save_interval == 0:
            torch.save(self.model.state_dict(), f"./saves/loc_ckpt_%d.pth" % epoch)


def main():
    # argparse
    parser = argparse.ArgumentParser(description="U-Net parameter selection")
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

    # dataset

    dataset_path = './datasets/localization/'
    trainset, validset, testset = None, None, None
    dataset = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }
    for name in ['train', 'valid', 'test']:
        set_path = os.path.join(dataset_path, f"{name}.txt")
        dataset[name] = SpineLocDataset(list_path=set_path)

    # model
    model = ResUnet(in_channels=cfg.IN_CH, out_channels=cfg.SEG.OUT_CH, init_features=cfg.INIT_FEATURES,
                    num_anchors=3, num_classes=6)

    # Training
    trainer = Trainer(args, dataset, model)
    trainer.train()


if __name__ == '__main__':
    main()

