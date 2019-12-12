
# Unet-multitask
A minimal implementation of unet-based model, aimed to complete segmentation and localization with hard shared parameters.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/linchenq/unet-segmentation-lumbar
    $ cd unet-multitask
    $ sudo pip3 install -r requirements.txt

##### Dataset preparation
Assume that the dataset is well created as the following formats:
> Each matrix, saved as `.mat`, contains original and masked images with specific channels. These channels represent vetebrates, disks, relevant regions and others respectively. Attention, the saved original
images is equal to 2-D numpy thus have to be enlarged with required input channels.

Then it's highly recommended to modify ```__C.SEG.FOLDER```, ```ROOT_PATH``` respectively in ```cfgs/config.py```and```datasets/localization/gen_data.py``` as your configuration.
 
#####  Custom dataset generation
	$ python datasets/localization/gen_data.py
	$ python datasets/segmentation/gen_data.py

## Task description
This task contains baseline, segmentation and localization. Each one's training and evaluating phase are based on the similar wheel. It's recommend to use the following command to help you set parameters or modify the default settings in ```cfgs/config.py```

	$ python train.py -h


## Train

#### Example 

#### Training log

#### Tensorboard


## Credit