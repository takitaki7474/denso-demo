import os
import glob
from PIL import Image
import torchvision.transforms as transforms


def load_dataset(dataset_dir, imsize=(32,32), img_name=False):
    dataset = []
    num_label = 2
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    for label in range(num_label):
        path = os.path.join(dataset_dir, str(label), "*.jpg")
        files = glob.glob(path)
        for file in files:
            img = Image.open(file)
            img = loader(img)
            if img_name:
                img_name = file.split("/")[-1]
                dataset.append((img, label, img_name))
            else:
                dataset.append((img, label))
    return dataset
