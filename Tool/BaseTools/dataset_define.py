import os.path

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from Tool.BaseTools.dataaugmentation import SSDAugmentation, BaseAugmentation
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.transforms as transforms
import numpy as np
from .cv2_ import CV2
from PIL import Image
from typing import List, Union


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


class VOCDataSet(Dataset):
    def __init__(
            self,
            root: str,
            years: list,
            train: bool = True,
            image_size: tuple = (448, 448),
            transform: Union[SSDAugmentation, BaseAugmentation] = None,
    ):
        # .../VOC/year/trainval(or test)/ ----
        super().__init__()
        self.root = root
        self.years = years
        self.train = train
        if self.train:
            self.data_type = 'trainval'
        else:
            self.data_type = 'test'
        self.image_size = image_size

        if transform is None:
            self.transform = BaseAugmentation()
        else:
            self.transform = transform

        self.image_and_xml_path_info = self.__get_image_and_xml_file_abs_path()

    def __get_image_and_xml_file_abs_path(self) -> list:
        res = []
        if self.train:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
                )
                txt_file_name = os.path.join(
                    root_path,
                    'ImageSets',
                    'Main',
                    '{}.txt'.format(self.data_type)
                )
                with open(txt_file_name, 'r') as f:
                    temp = f.readlines()
                    xml_file_names = [val[:-1] + '.xml' for val in temp]

                res += [(root_path, xml_file_name) for xml_file_name in xml_file_names]
        else:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
                )
                anno_path = os.path.join(
                    root_path,
                    'Annotations'
                )
                xml_file_names = os.listdir(anno_path)
                res += [(root_path, xml_file_name) for xml_file_name in xml_file_names]
        return res

    def __len__(self):
        return len(self.image_and_xml_path_info)

    def __get_image_label(
            self,
            index,
    ) -> tuple:
        root_path, xml_file_name = self.image_and_xml_path_info[index]
        xml_trans = XMLTranslate(root_path=root_path, file_name=xml_file_name)
        # xml_trans.resize(new_size=self.image_size)
        return xml_trans.img, xml_trans.objects

    def __getitem__(self, index):
        img, label = self.__get_image_label(index)
        boxes = []
        classes = []

        for val in label:
            classes.append(val[0])
            boxes.append(val[1: 5])

        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes)
        new_img_tensor, new_boxes, new_classes = self.transform(
            img,
            boxes,
            classes
        )
        new_label = []
        for i in range(new_classes.shape[0]):
            new_label.append(
                (new_classes[i], *new_boxes[i].tolist())
            )
        return new_img_tensor, new_label

    @staticmethod
    def collate_fn(batch):
        # batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
        batch = list(zip(*batch))
        imgs = batch[0]
        labels = batch[1]
        del batch
        return torch.stack(imgs), labels


class VOCTrainValDataSet(Dataset):
    def __init__(
            self,
            root: str,
            year: str,
            train: bool = True,
            image_size: tuple = (448, 448),
            transform: torchvision.transforms.Compose = None,
            fast_load: bool = False,
    ):
        super().__init__()
        assert year in ['2012', '2007']
        self.train = train
        self.data_type = 'trainval'
        self.data_path = os.path.join(root, year, self.data_type)
        self.image_size = image_size
        self.transform = transform
        self.xml_file_names = self.__get_image_xml_file_names()

        self.fast_load = fast_load
        self.images = []
        self.labels = []
        if self.fast_load:
            for xml_name in self.xml_file_names:
                a, b = self.__get_image_label(xml_name)
                self.images.append(a)
                self.labels.append(b)

    def __get_image_xml_file_names(self) -> list:
        txt_file_name = os.path.join(
            self.data_path,
            'ImageSets',
            'Main',
            '{}.txt'.format(self.data_type)
        )

        with open(txt_file_name, 'r') as f:
            temp = f.readlines()
            xml_file_names = [val[:-1] + '.xml' for val in temp]

        n = len(xml_file_names)
        cut = int(0.8 * n)
        if self.train:
            return xml_file_names[0: cut]
        else:
            return xml_file_names[cut:]

    def __len__(self):
        return len(self.xml_file_names)

    def __get_image_label(
            self,
            xml_file_name: str,
    ) -> tuple:
        xml_trans = XMLTranslate(
            root_path=self.data_path,
            file_name=xml_file_name
        )
        xml_trans.resize(new_size=self.image_size)
        return xml_trans.img, xml_trans.objects

    def __getitem__(self, index):
        if self.fast_load:
            img, label = self.images[index], self.labels[index]
        else:
            img, label = self.__get_image_label(self.xml_file_names[index])

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

#######################################################


def get_voc_trainval_data_loader(
        root_path: str,
        year: str,
        image_size: tuple,
        batch_size: int,
        train: bool = True,
        num_workers: int = 0,
        fast_load: bool = False
):

    normalize = transforms.Normalize(
        std=[0.5, 0.5, 0.5],
        mean=[0.5, 0.5, 0.5],
    )
    if train:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_d = VOCTrainValDataSet(
            root=root_path,
            year=year,
            train=True,
            image_size=image_size,
            transform=transform_train,
            fast_load=fast_load,
        )

        train_l = DataLoader(
            train_d,
            batch_size=batch_size,
            collate_fn=VOCTrainValDataSet.collate_fn,
            shuffle=True,
            num_workers=num_workers
                             )
        return train_l
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        test_d = VOCTrainValDataSet(
            root=root_path,
            year=year,
            train=False,
            image_size=image_size,
            transform=transform_test,
            fast_load=fast_load,
        )

        test_l = DataLoader(
            test_d,
            batch_size=batch_size,
            collate_fn=VOCTrainValDataSet.collate_fn,
            shuffle=False,
            num_workers=num_workers
        )
        return test_l


def get_voc_data_loader(
        root_path: str,
        years: list,
        image_size: tuple,
        batch_size: int,
        train: bool = True,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
):

    if train:
        transform_train = SSDAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
        )

        train_d = VOCDataSet(
            root=root_path,
            years=years,
            train=True,
            image_size=image_size,
            transform=transform_train
        )

        train_l = DataLoader(train_d,
                             batch_size=batch_size,
                             collate_fn=VOCDataSet.collate_fn,
                             shuffle=True)
        return train_l
    else:
        transform_test = BaseAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
        )

        test_d = VOCDataSet(
            root=root_path,
            years=years,
            train=False,
            image_size=image_size,
            transform=transform_test
        )

        test_l = DataLoader(test_d,
                            batch_size=batch_size,
                            collate_fn=VOCDataSet.collate_fn,
                            shuffle=False)
        return test_l


def get_image_net_224_loader(
        root_path: str,
        batch_size: int,
        train: bool = True,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = get_imagenet_dataset(
            root=root_path,
            transform=transform_train,
            train=True
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader

    else:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_set = get_imagenet_dataset(
            root=root_path,
            transform=transform_test,
            train=False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
        )
        return test_loader


def get_image_net_448_loader(
        root_path: str,
        batch_size: int,
        train: bool = True,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,

        ])

        train_set = get_imagenet_dataset(
            root=root_path,
            transform=transform_train,
            train=True
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader
    else:
        transform_test = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize
        ])

        test_set = get_imagenet_dataset(
            root=root_path,
            transform=transform_test,
            train=False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
        )
        return test_loader
