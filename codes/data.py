import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, img_root, gt_root, trainsize):
        self.trainsize = trainsize

        self.images = [img_root + f for f in os.listdir(img_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize * 2)),
            transforms.ToTensor()])

        self.img_transform_er_cube = transforms.Compose([
            transforms.Resize((640, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform_er_cube = transforms.Compose([
            transforms.Resize((640, 1280)),
            transforms.ToTensor()])

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        ER_image = self.img_transform_er_cube(image)
        image = self.img_transform(image)

        ER_gt = self.gt_transform_er_cube(gt)
        gt = self.gt_transform(gt)

        return image, gt, ER_image, ER_gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(img_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(img_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


#test dataset and loader
class test_dataset:
    def __init__(self, img_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [img_root + f for f in os.listdir(img_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
       # self.gt_transform = transforms.ToTensor()
       # self.gt_transform = transforms.Compose([
       #     transforms.Resize((self.testsize, self.testsize))])

        self.transform_er_cube = transforms.Compose([
            transforms.Resize((640, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        ER_img = self.transform_er_cube(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = gt.resize((self.testsize * 2, self.testsize))
        name = self.gts[self.index].split('/')[-1]
        #if name.endswith('.jpg'):
        #    name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, name, ER_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class test_dataset_pano:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize * 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #self.gt_transform = transforms.ToTensor()

        self.transform_er_cube = transforms.Compose([
            transforms.Resize((640, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        ER_img = self.transform_er_cube(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.gts[self.index].split('/')[-1]
        self.index += 1
        self.index = self.index % self.size
        return image, gt, ER_img, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class test_dataset_2D:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.gts[self.index].split('/')[-1]
        self.index += 1
        self.index = self.index % self.size
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
