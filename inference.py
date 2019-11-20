import os
import torch
from terminaltables import AsciiTable
import matplotlib.pyplot as plt

from cfgs.config import cfg
from utils import *

class Inference(object):
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataloader = {
            'test': DataLoader(dataset['test'], batch_size=self.args.batch_size, shuffle=True),
        }
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def inference(self):
        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model.eval()

        for batch_i, (x, y_test) in enumerate(self.dataloader['test']):
            x, y_test = x.float(), y_test.float()
            x, y_test = x.to(self.device), y_test.to(self.device)

            y_pred = self.model(x)
            n_classes = cfg.SEG.OUT_CH

            # ground truth
            fig, axs = plt.subplots(nrows=1, ncols=n_classes)
            for i in range(n_classes):
                axs[i].imshow(y_test.data.cpu().numpy()[0, i], cmap='gray')
            # pred
            fig, axs = plt.subplots(nrows=1, ncols=n_classes)
            for i in range(n_classes):
                axs[i].imshow(y_pred.data.cpu().numpy()[0, i], cmap='gray')

            # dice metric
            metric_k, metric_v = range(n_classes), []
            for i in range(n_classes):
                metric_v.append(dice_coeff(y_test.data.cpu().numpy()[0, i], y_pred.data.cpu().numpy()[0, i]))
            print (AsciiTable([metric_k, metric_v]).table)

def main():
    # argparse
    parser = argparse.ArgumentParser(description="U-Net parameter selection")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=cfg.TRAIN.DEVICE)
    parser.add_argument("--model_path", type=str, default="./saves/unet_ckpt_140.pth")
    args = parser.parse_args()

    # asd = "./saves/unet_ckpt_30.pth"
    # print(os.path.exists(asd))

    # dataset
    dataset_path = './datasets/segmentation/'
    testset = None
    dataset = {
        'test': testset
    }
    dataset['test'] = SpineSegDataset(list_path=os.path.join(dataset_path, "test.txt"))

    # model
    model = ResUnet(in_channels=cfg.IN_CH, out_channels=cfg.SEG.OUT_CH, init_features=cfg.INIT_FEATURES)
    # Training
    inference = Inference(args, dataset, model)
    inference.inference()


if __name__ == '__main__':
    main()