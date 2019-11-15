from collections import defaultdict
import torch

from utils.loss import *

def evaluate(model, dataset):
    model.eval()

    metrics = defaultdict(float)
    unetloss = UnetLoss()
    samples = 0
    loss_valid = []

    for batch_i, (x, y_true) in enumerate(dataset['valid']):
        x, y_true = x.float(), y_true.float()
        x, y_true = x.to(self.device), y_true.to(self.device)

        with torch.no_grad():
            y_pred = model(x)
            _loss = unetloss.forward(y_pred, y_true, metrics=metrics)
            loss_valid.append(_loss.item())

        samples += x.size(0)

    return loss_valid, metrics