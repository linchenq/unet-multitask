import os
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True)
        }

        # prerequisites
        os.makedirs(self.args.log_path, exist_ok=True)
        os.makedirs(self.args.save_path, exist_ok=True)

        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.loss = UnetLoss()

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_scheduler_step, gamma=0.1)

        self.model.to(self.device)

        self.logger = Logger(self.args.log_path)

    def train(self):
        num_epochs = self.args.epoch

        for epoch in range(num_epochs):
            log_str = "-"*5 + f" [Epoch {epoch}/{num_epochs-1}] " + "-"*5
            print(log_str)
            since = time.time()

            self.run_single_step(epoch)
            self.exp_lr_scheduler.step()

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def run_single_step(self, epoch):
        self.model.train()

        metrics = defaultdict(float)
        samples = 0
        loss_train = []

        for batch_i, (x, y_true) in enumerate(self.dataloader['train']):
            log_str = "---- [Phase %s][Epoch %d/%d, Batch %d/%d] ----"% \
                        ('train', epoch, self.args.epoch, batch_i, len(self.dataloader['train']))
            print(log_str)
            x, y_true = x.float(), y_true.float()
            x, y_true = x.to(self.device), y_true.to(self.device)

            self.optimizer.zero_grad()

            # forward & backward
            with torch.enable_grad():
                y_pred = self.model(x)
                _loss = self.loss.forward(y_pred, y_true, metrics=metrics)
                loss_train.append(_loss.item())
                _loss.backward()
                self.optimizer.step()

            samples += x.size(0)

        print_metrics(metrics, samples, 'train', epoch)
        epoch_loss = metrics['loss'] / samples

        self.logger.scalar_summary('loss/train', np.mean(loss_train), epoch)

        if epoch % self.args.eval_interval == 0:
            valid_loss, valid_metrics, valid_samples = evaluate(self.model, self.dataloader, self.device)
            print_metrics(valid_metrics, valid_samples, 'valid', epoch)
            self.logger.scalar_summary('loss/valid', np.mean(valid_loss), epoch)
        if epoch % self.args.save_interval == 0:
            torch.save(self.model.state_dict(), f"./saves/unet_ckpt_%d.pth" % epoch)



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

    parser.add_argument("--device", type=str, default=cfg.TRAIN.DEVICE)
    args = parser.parse_args()

    # dataset
    dataset_path = './datasets/segmentation/'
    trainset, validset, testset = None, None, None
    dataset = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }
    for name in ['train', 'valid', 'test']:
        set_path = os.path.join(dataset_path, f"{name}.txt")
        dataset[name] = SpineSegDataset(list_path=set_path)

    # model
    model = ResUnet(in_channels=cfg.IN_CH, out_channels=cfg.SEG.OUT_CH, init_features=cfg.INIT_FEATURES)
    # Training
    trainer = Trainer(args, dataset, model)
    trainer.train()


if __name__ == '__main__':
    main()



















