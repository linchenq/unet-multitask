import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *

class SuperYoloLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(SuperYoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.metrics = {}

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100

        self.grid_size = 0

    # Based on:
    # --------------------------------------------------------
    # PyTorch-YOLOv3
    # Copyright @eriklindernoren
    # Licensed under GNU General Public License v3.0
    # --------------------------------------------------------

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def generate_output(self, x, target=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output, x, y, w, h, pred_conf, pred_cls, pred_boxes

    def forward(x, targets):
        raise NotImplementedError


class YoloLoss(SuperYoloLayer):

    def forward(self, x, targets=None):
        output, x, y, w, h, pred_conf, pred_cls, pred_boxes = self.generate_output(x, targets)

        if targets is None:
            raise ValueError("target cannot be none when computing losses")

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )

        # convert obj_mask, noobj_mask from bytetensor to bool
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "grid_size": self.grid_size,
            "loss": total_loss.detach().cpu().item(),
            "x": loss_x.detach().cpu().item(),
            "y": loss_y.detach().cpu().item(),
            "w": loss_w.detach().cpu().item(),
            "h": loss_h.detach().cpu().item(),
            "conf": loss_conf.detach().cpu().item(),
            "cls": loss_cls.detach().cpu().item(),
            "cls_acc": cls_acc.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_noobj": conf_noobj.detach().cpu().item(),
        }

        return total_loss


class YoloLayer(SuperYoloLayer):
    def forward(self, x):
        output, x, y, w, h, pred_conf, pred_cls, pred_boxes = self.generate_output(x)
        return output


class SegLoss(nn.Module):
    @staticmethod
    def _dice_loss(y_pred, y_true, smooth):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
        dice_loss = (1 - ((2. * intersection + smooth) / (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + smooth)))
        dice_loss = dice_loss.mean()
        return dice_loss

    @staticmethod
    def _bce_loss(y_pred, y_true):
        bce_loss = nn.BCEWithLogitsLoss()
        bce_loss = bce_loss(y_pred, y_true)
        return bce_loss

    def forward(self, y_pred, y_true, metrics, mode="both"):
        dice = self._dice_loss(y_pred, y_true, self.smooth)
        bce = self._bce_loss(y_pred, y_true)
        metrics['dice_seg'] += dice.data.cpu().numpy() * y_true.size(0)
        metrics['bce_seg'] += bce.data.cpu().numpy() * y_true.size(0)

        bce_weight = 0.5
        bce_dice_loss = bce_weight * bce + (1. - bce_weight) * dice
        metrics['both_seg'] = bce_dice_loss.data.cpu().numpy() * y_true.size(0)

        if mode == "bce":
            metrics["seg_loss"] = metrics['bce_seg']
            return bce
        elif mode == "dice":
            metrics["seg_loss"] = metrics['dice_seg']
            return dice
        elif mode == "both":
            metrics["seg_loss"] = metrics['both_seg']
            return bce_dice_loss
        else:
            raise NotImplementedError
