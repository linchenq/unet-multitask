# unet-multitask

## Segmentation

### Data preparation

#### Data format
Each matrix, saved as `.mat`, contains original and masked images with specific channels. These channels
represent vetebrates, disks, relevant regions and others respectively. Attention, the saved original
images is equal to 2-D numpy thus have to be enlarged with required input channels.