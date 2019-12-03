import numpy as np
from terminaltables import AsciiTable

def yolo_metrics(epoch, phase, metrics):
    index = ['Metrics', 'Yolo Layer 3', 'Yolo Layer 2', 'Yolo Layer 1']
    epoch_row = ['Epoch', epoch, epoch, epoch]
    phase_row = ['Phase', phase, phase, phase]

    res_table = [index, epoch_row, phase_row]
    for k, v in metrics.items():
        res_table.append([k] + v)
    return AsciiTable(res_table).table


def dice_coeff(im1, im2):
    # im1 = np.asarray(im1).astype(np.bool)
    # im2 = np.asarray(im2).astype(np.bool)
    im1 = im1 > 0.5
    im2 = im2 > 0.5

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())