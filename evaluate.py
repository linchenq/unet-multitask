from collections import defaultdict
import torch

from cfgs.config import cfg
from utils.loss import *
from utils.metrics import *

def evaluate(model, dataset, device, epoch, logger):
    model.eval()

    eval_dataset = dataset['valid'] if 'valid' in dataset else dataset
    for batch_i, (x, targets) in enumerate(eval_dataset):
        x, targets = x.float(), targets.float()
        x, targets = x.to(self.device), targets.to(self.device)

        losses = []
        num_anchors = cfg.LOC.NUM_ANCHORS
        for index in range(int(len(cfg.LOC.ANCHORS) / num_anchors)):
            losses.append(YoloLoss(cfg.LOC.ANCHORS[index*num_anchors: (index+1)*num_anchors],
                                      cfg.LOC.NUM_CLASSES,
                                      img_size=cfg.H))

        epoch_metrics = {
                "grid_size": [],
                "total_loss": [],
                "x": [], "y": [], "w": [], "h": [],
                "conf": [], "cls": [],
                "cls_acc": [],
                "recall50": [], "recall75": [], "precision": [],
                "conf_obj": [], "conf_noobj": []
        }

        with torch.no_grad():
            yolo = model(x)
            for i in range(len(yolo)):
                i_loss = losses[i].forward(yolo[i], targets)
                for k, v in self.loss[i].metrics.items():
                    epoch_metrics[k].append(v)

        # metrics
        out_metric = yolo_metrics(epoch=epoch, phase="valid", metrics=epoch_metrics)
        print(out_metric)
        logger.log_summary(mode="INFO", msg=out_metric)

        # tensorboard
        vis_step = len(eval_dataset) * epoch + batch_i
        vis_metrics = [("train/total_loss", np.array(epoch_metrics["total_loss"]).sum()),
                       ("train/x_loss", np.array(epoch_metrics["x"]).sum()),
                       ("train/y_loss", np.array(epoch_metrics["y"]).sum()),
                       ("train/w_loss", np.array(epoch_metrics["w"]).sum()),
                       ("train/h_loss", np.array(epoch_metrics["h"]).sum())]
        self.logger.list_of_scalars_summary(vis_metrics, vis_step)


# def evaluate(model, dataset, device):
#     model.eval()

#     metrics = defaultdict(float)
#     unetloss = UnetLoss()
#     samples = 0
#     loss_valid = []

#     for batch_i, (x, y_true) in enumerate(dataset['valid']):
#         x, y_true = x.float(), y_true.float()
#         x, y_true = x.to(device), y_true.to(device)

#         with torch.no_grad():
#             y_pred = model(x)
#             _loss = unetloss.forward(y_pred, y_true, metrics=metrics)
#             loss_valid.append(_loss.item())

#         samples += x.size(0)

#     return loss_valid, metrics, samples