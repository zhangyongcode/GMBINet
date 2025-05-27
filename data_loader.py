from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import os


# ==========================dataset load==========================

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'image': img, 'label': lbl, 'name': name}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label, 'name': name}


class ToTensor(object):
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:

            label = label
        else:
            label = label / np.max(label)

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.4669) / 0.2437
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.4669) / 0.2437
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.4669) / 0.2437
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.4669) / 0.2437

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),
                'label': torch.from_numpy(tmpLbl),
                'name': name}


class SalObjDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_ext='.bmp', label_ext='.png'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.data_list = []
        self._get_image_list()

        self.transform = transform

    def _get_image_list(self):
        image_dir, label_dir = self.image_dir, self.label_dir

        files = os.listdir(image_dir)
        for f in files:
            image_path = os.path.join(image_dir, f)
            label_name = f.replace(self.image_ext, self.label_ext)
            name = f[0:-4]
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(image_path) and os.path.exists(label_path):

                # 直接将数据缓存内存中，便于处理
                self.data_list.append([image_path, label_path, name])
            else:
                # 当不能提供label
                self.data_list.append([image_path, image_path, name])
        print(len(files), 'are selected')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label_path, name = self.data_list[idx]
        image = io.imread(image_path)
        label_3 = io.imread(label_path)

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]
        label = np.array(label, dtype=np.float32)
        sample = {'image': image, 'label': label, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample
