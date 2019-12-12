import os
import argparse
import time
import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cfgs.config import cfg
from baseline import *
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
        # self.dataloader = {
        #     'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset['train'].collate_fn),
        #     'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True, collate_fn=dataset['valid'].collate_fn)
        # }
        self.dataloader = {
            'train': DataLoader(dataset['train'], batch_size=self.args.batch_size, shuffle=True),
            'valid': DataLoader(dataset['valid'], batch_size=self.args.batch_size, shuffle=True)
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
        self.loss = SegLoss()

        # optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=self.args.lr_scheduler_step,
                                                          gamma=0.1)

    def train(self):
        num_epochs = self.args.epoch

        for epoch in tqdm.tqdm(range(num_epochs)):
            since = time.time()

            self.run_single_step(epoch)
            self.exp_lr_scheduler.step()

            time_elapsed = time.time() - since
            print("Epoch {}: {:.0f}m {:.0f}s".format(epoch, time_elapsed // 60, time_elapsed % 60))
            self.logger.log_summary(mode="INFO", msg="Epoch {}: {:.0f}m {:.0f}s".format(epoch, time_elapsed // 60, time_elapsed % 60))

    def run_single_step(self, epoch):
        self.model.train()
        
        epoch_metrics = {
            "seg_loss": [],
            "dice_loss": [], "bce_loss": [], "db_loss": []
        }

        for batch_i, (x, masks) in enumerate(self.dataloader['train']):
            x, masks = x.float(), masks.float()
            x, masks = x.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.enable_grad():
                seg_res = self.model(x)                
                seg_loss = self.loss.forward(seg_res, masks)
                # metrics logger
                for k, v in self.loss.metrics.items():
                    epoch_metrics[k].append(v)
            
                seg_loss.backward()
                self.optimizer.step()
        
        #TODO
        # DEBUG TO TEST IF EPOCH_METRICS IS TENSOR / NUMPY?
        
        # metrics logger
        vis_metrics = [("train/seg_loss", np.array(epoch_metrics["seg_loss"]).sum()),
                       ("train/dice_loss", np.array(epoch_metrics["dice_loss"]).sum()),
                       ("train/bce_loss", np.array(epoch_metrics["bce_loss"]).sum()),
                       ("train/db_loss", np.array(epoch_metrics["db_loss"]).sum())]
        self.logger.list_of_scalars_summary(vis_metrics, epoch)
        
        for k, v in epoch_metrics.items():
            epoch_metrics[k] = np.array(v).sum()
            
        epoch_metrics = {**{"epoch": epoch, "phase":"train"}, **epoch_metrics}
        out_metrics = general_metrics(epoch_metrics)
        print(out_metrics)
        self.logger.log_summary(mode="INFO", msg=out_metrics)

        if epoch % self.args.eval_interval == 0:
            # evaluate(model=self.model, dataset=self.dataloader, device=self.device,
            #          iou_thres=0.5, conf_thres=0.001, nms_thres=0.5, img_size=cfg.H,
            #          epoch=epoch, logger=self.logger)
            pass

        if epoch % self.args.save_interval == 0:
            torch.save(self.model.state_dict(), f"./saves/seg_ckpt_%d.pth" % epoch)


def main():
    # argparse
    parser = argparse.ArgumentParser(description="Baseline Segmentation")
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
    dataset_path = '../datasets/segmentation/'
    trainset, validset, testset = None, None, None
    dataset = {
        'train': trainset,
        'valid': validset,
    }
    for name in ['train', 'valid']:
        set_path = os.path.join(dataset_path, f"{name}.txt")
        dataset[name] = SpineSegDataset(list_path=set_path)

    # model
    model = Baseline(in_channels=cfg.IN_CH,
                     out_channels=cfg.SEG.OUT_CH,
                     init_features=cfg.INIT_FEATURES)

    # Training
    trainer = Trainer(args, dataset, model)
    trainer.train()


if __name__ == '__main__':
    main()


