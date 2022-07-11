from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from UTILS.cv2_ import CV2
from PIL import Image


class XMLTranslate:
    def __init__(self, root_path: str, file_name: str):
        if not root_path.endswith('/'):
            root_path += '/'

        self.annotations_path = root_path + 'Annotations/'
        self.images_path = root_path + 'JPEGImages/'

        self.root = ET.parse(self.annotations_path + file_name).getroot()  # type: xml.etree.ElementTree.Element
        self.img = None   # type:np.ndarray
        self.img_file_name = None  # type:str
        self.img_size = None  # type:tuple
        self.objects = []  # type:list
        self.__set_info()

    def __set_info(self):

        self.img_file_name = self.root.find('filename').text.strip()
        self.img = CV2.imread(self.images_path + self.img_file_name)

        for size in self.root.iter('size'):
            img_w = float(size.find('width').text.strip())
            img_h = float(size.find('height').text.strip())
            img_c = int(size.find('depth').text.strip())
            self.img_size = (img_w, img_h, img_c)

        for obj in self.root.iter('object'):
            kind = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            a = float(bbox.find('xmin').text.strip())  # point to x_axis dist
            b = float(bbox.find('ymin').text.strip())  # point to y_axis dist
            m = float(bbox.find('xmax').text.strip())
            n = float(bbox.find('ymax').text.strip())
            self.objects.append((kind, a, b, m, n))

    def resize(self, new_size: tuple = (448, 448)):
        self.img = CV2.resize(self.img, new_size)
        old_w = self.img_size[0]
        old_h = self.img_size[1]
        new_w = new_size[0]
        new_h = new_size[1]
        self.img_size = (new_w, new_h, self.img_size[2])

        for i in range(len(self.objects)):
            new_object = (
                self.objects[i][0],
                self.objects[i][1] / (old_w / new_w),
                self.objects[i][2] / (old_h / new_h),
                self.objects[i][3] / (old_w / new_w),
                self.objects[i][4] / (old_h / new_h)
            )
            self.objects[i] = new_object

    def get_image_size(self) -> tuple:
        return self.img_size

    def get_image_name(self) -> str:
        return self.img_file_name

    def get_objects(self) -> list:
        return self.objects

    def print(self):
        print("image name: {}".format(self.img_file_name))
        print("image size: {}".format(self.img_size))
        print("objects:")
        for val in self.objects:
            print("kind: {}, box: ({},{},{},{})".format(val[0], val[1], val[2], val[3], val[4]))


class VOC2012DataSet(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 image_size: tuple = (448, 448),
                 transform: torchvision.transforms.Compose = None):
        super().__init__()

        self.root = root
        if not self.root.endswith('/'):
            self.root += '/'

        self.train = train
        self.image_size = image_size
        self.transform = transform
        self.images, self.labels = self.__get_images_and_labels()

    def __get_images_and_labels(self) -> tuple:
        images = []
        labels = []

        train_or_val_txt_file_name = self.root + 'ImageSets/Main/' + 'trainval.txt'

        with open(train_or_val_txt_file_name, 'r') as f:
            temp = f.readlines()
            xml_file_names = [val[:-1]+'.xml' for val in temp]

        for val in xml_file_names:
            xml_trans = XMLTranslate(root_path=self.root, file_name=val)
            xml_trans.resize(new_size=self.image_size)
            images.append(xml_trans.img)
            labels.append(xml_trans.objects)
        total = len(images)
        cut = int(0.7*total)
        if self.train:
            return images[0: cut], labels[0: cut]
        else:
            return images[cut: ], labels[cut: ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.images[index]
        img = CV2.cvtColorToRGB(img)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = img / 255.0
        return img, label

    @staticmethod
    def collate_fn(batch):
        # batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
        batch = list(zip(*batch))
        imgs = batch[0]
        labels = batch[1]
        del batch
        return torch.stack(imgs), labels

def get_imagenet_dataset(
        root: str,
        transform: Compose,
        train: bool = True,
):
    if train:
        path = '{}/train/'.format(root)
    else:
        path = '{}/val/'.format(root)

    return ImageFolder(path, transform)

