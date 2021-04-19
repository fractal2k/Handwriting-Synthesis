import os
import cv2
import config
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Define global constants
src_transforms = transforms.Compose(
    [
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)


class HandwritingDataset(Dataset):
    """Custom Dataset that returns (Image, Label) pairs"""

    def __init__(self, main_dir, transforms):
        self.main_dir = main_dir
        self.transforms = transforms
        self.all_imgs = os.listdir(self.main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        # image = cv2.imread(img_loc, 0)
        # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # image = Image.fromarray(image)
        image = Image.open(img_loc).convert("L")
        tensor_image = self.transforms(image)
        label = self.all_imgs[idx].split(" ")[0].split(".")[0]

        return tensor_image, label, len(label)


def get_dataloader():
    """Convenience function that returns dataset and dataloader objects"""
    trainset = HandwritingDataset(config.SRC_DIR, src_transforms)
    trainloader = DataLoader(
        trainset, config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4
    )

    return trainset, trainloader
