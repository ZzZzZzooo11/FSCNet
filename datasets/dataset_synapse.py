import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def resize_(image, label, output_size):
    w = output_size[0]
    h = output_size[1]

    iw, ih = image.size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', [w, h], (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', [w, h], (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

    return new_image, new_label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label




class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = resize_(image, label, self.output_size)

        image = np.array(image)
        label = np.array(label)

        image = image.transpose((2, 0, 1))


        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split

        list_file = os.path.join(list_dir, self.split + '.txt').replace('\\', '/')
        with open(list_file, 'r') as f:
            self.sample_list = f.readlines()
        self.data_dir = base_dir.replace('\\', '/')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')

        img_dir = os.path.join(self.data_dir, "images")

        image_path = os.path.join(img_dir, slice_name + '.jpg').replace('\\', '/')
        if not os.path.exists(image_path):
            image_path = os.path.join(img_dir, slice_name + '.png').replace('\\', '/')
        if not os.path.exists(image_path):
            image_path = os.path.join(img_dir, slice_name + '.jpeg').replace('\\', '/')

        label_path = os.path.join(self.data_dir, "masks", slice_name + '.png').replace('\\', '/')

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found (checked .jpg/.png/.jpeg): {os.path.join(img_dir, slice_name)}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        label = np.array(label)
        label[label > 0] = 1
        label = Image.fromarray(np.uint8(label))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample