from collections import defaultdict
import numpy as np
import torch
import tqdm

from cfgs.config import cfg
from utils.loss import *
from utils.metrics import *

def evaluate(model, dataset, device,
             iou_thres, conf_thres, nms_thres, img_size,
             epoch, logger):

    model.eval()

    # Yolo layers
    losses, yolo_layer = [], []
    num_anchors = cfg.LOC.NUM_ANCHORS
    for index in range(int(len(cfg.LOC.ANCHORS) / num_anchors)):
        losses.append(YoloLoss(cfg.LOC.ANCHORS[index*num_anchors: (index+1)*num_anchors],
                                  cfg.LOC.NUM_CLASSES,
                                  img_size=cfg.H))
        yolo_layer.append(YoloLayer(cfg.LOC.ANCHORS[index*num_anchors: (index+1)*num_anchors],
                                    cfg.LOC.NUM_CLASSES,
                                    img_size=cfg.H))

    # Metrics for detection
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    eval_dataset = dataset['valid'] if 'valid' in dataset else dataset
    for batch_i, (x, targets) in enumerate(tqdm.tqdm(eval_dataset, desc="Detecting objects")):
        x, targets = x.float(), targets.float()
        x, targets = x.to(device), targets.to(device)

        epoch_metrics = {
                "grid_size": [],
                "local_loss": [],
                "x": [], "y": [], "w": [], "h": [],
                "conf": [], "cls": [],
                "cls_acc": [],
                "recall50": [], "recall75": [], "precision": [],
                "conf_obj": [], "conf_noobj": []
        }
        epoch_format = {
            "grid_size": "%2d",
            "local_loss": "%.6f",
            "x": "%.6f", "y": "%.6f", "w": "%.6f", "h": "%.6f",
            "conf": "%.6f", "cls": "%.6f",
            "cls_acc": "%.2f%%",
            "recall50": "%.6f", "recall75": "%.6f", "precision": "%.6f",
            "conf_obj": "%.6f", "conf_noobj": "%.6f"
        }

        with torch.no_grad():
            output = model(x)
            yolo = output[:3]
            i_yolo = []

            for i in range(len(yolo)):
                # YOLO LOSS
                i_loss = losses[i].forward(yolo[i], targets)
                for k, v in losses[i].metrics.items():
                    epoch_metrics[k].append(v)

                # YOLO OUTPUT FOR DETECTION
                i_yolo.append(yolo_layer[i].forward(yolo[i]))

            i_yolo = torch.cat(i_yolo, 1)
            detection = non_max_suppression(i_yolo, conf_thres, nms_thres)

            # Extract labels & Rescale target
            labels += targets[:, 1].tolist()
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            sample_metrics += get_batch_statistics(detection, targets, iou_thres)

        # metrics for validation phase
        out_table = yolo_metrics(epoch=epoch, phase="valid", metrics=epoch_metrics, formats=epoch_format)
        print(out_table)
        logger.log_summary(mode="INFO", msg=out_table)

        # tensorboard
        vis_step = len(eval_dataset) * epoch + batch_i
        vis_metrics = [("valid/local_loss", np.array(epoch_metrics["local_loss"]).sum()),
                       ("valid/x_loss", np.array(epoch_metrics["x"]).sum()),
                       ("valid/y_loss", np.array(epoch_metrics["y"]).sum()),
                       ("valid/w_loss", np.array(epoch_metrics["w"]).sum()),
                       ("valid/h_loss", np.array(epoch_metrics["h"]).sum())]
        logger.list_of_scalars_summary(vis_metrics, vis_step)

    # metrics for detection
    if sample_metrics:
        true_positives = np.concatenate(list(zip(*sample_metrics))[0], 0)
        pred_scores = np.concatenate([x.cpu() for x in list(zip(*sample_metrics))[1]], 0)
        pred_labels = np.concatenate([x.cpu() for x in list(zip(*sample_metrics))[2]], 0)
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        precision, recall, AP, f1, ap_class = np.array([-1]), np.array([-1]), np.array([-1]), np.array([-1]), -1

    detection_table = general_metrics(metrics={"epoch": epoch,
                                               "phase": "valid",
                                               "val_precision": precision.mean(),
                                               "val_recall": recall.mean(),
                                               "val_mAP": AP.mean(),
                                               "val_f1": f1.mean()
                                               }
    )
    print(detection_table)
    logger.log_summary(mode="INFO", msg=detection_table)
    detection_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(detection_metrics, epoch)