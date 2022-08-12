from Tool.BaseTools import VOCDataSet, SSDAugmentation, BaseAugmentation, XMLTranslate, CV2
from typing import Union, List
import numpy as np
import random
from torch.utils.data import DataLoader


class StrongerVOCDataSet(VOCDataSet):
    def __init__(
            self,
            root: str,
            years: list,
            kinds_name: list,
            train: bool = True,
            image_size: tuple = (448, 448),
            transform: Union[SSDAugmentation, BaseAugmentation] = None,
            use_mosaic: bool = False,
            use_mixup: bool = False,

    ):
        super().__init__(root, years, train, image_size, transform)
        self.use_mixup = use_mixup
        self.use_mosaic = use_mosaic
        self.ids = [i for i in range(len(self.image_and_xml_path_info))]
        self.kinds_name = kinds_name

    def __get_origin_one(
            self,
            index
    ):
        root_path, xml_file_name = self.image_and_xml_path_info[index]
        xml_trans = XMLTranslate(root_path=root_path, file_name=xml_file_name)
        # xml_trans.resize(new_size=self.image_size)

        img, label = xml_trans.img, xml_trans.objects
        boxes = []
        classes = []

        for val in label:
            classes.append(val[0])
            boxes.append(val[1: 5])

        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes)

        return img, boxes, classes

    def __resize_image(
            self,
            image: np.ndarray,
            boxes: np.ndarray,
            new_size: tuple
    ):
        old_h, old_w, _ = image.shape
        new_image = CV2.resize(image, new_size)
        new_h, new_w, _ = new_image.shape
        new_boxes = boxes.copy()
        new_boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_w * new_w
        new_boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_h * new_h
        return new_image, new_boxes

    def __put_image_on_small_back_ground(
            self,
            image: np.ndarray,
            boxes: np.ndarray
    ):
        image, boxes = self.__resize_image(image, boxes, self.image_size)
        back_ground = np.zeros(shape=image.shape)

        back_ground_h, back_ground_w, _ = image.shape
        scaled_rate = 0.01 * np.random.randint(50, 100)

        new_w = int(scaled_rate * back_ground_w)
        new_h = int(scaled_rate * back_ground_h)

        image, boxes = self.__resize_image(image, boxes, (new_w, new_h))

        offset_x = np.random.randint(0, back_ground_w - new_w)
        offset_y = np.random.randint(0, back_ground_h - new_h)

        back_ground[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = image
        back_ground_boxes = boxes.copy()
        back_ground_boxes[:, [0, 2]] = boxes[:, [0, 2]] + offset_x
        back_ground_boxes[:, [1, 3]] = boxes[:, [1, 3]] + offset_y
        return back_ground, back_ground_boxes

    def __get_mosaic(
            self,
            index
    ):
        ids_list_ = self.ids[:index] + self.ids[index + 1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]
        offset_vec = [
            [0, 0],
            [self.image_size[0], 0],
            [0, self.image_size[1]],
            [self.image_size[0], self.image_size[1]],
        ]
        four_back_ground = np.zeros(shape=(self.image_size[1] * 2, self.image_size[0] * 2, 3))
        boxes_vec = []
        class_vec = []
        for i in range(len(ids)):
            id = ids[i]
            offset_x, offset_y = offset_vec[i]
            img, boxes, classes = self.__get_origin_one(id)
            img, boxes = self.__put_image_on_small_back_ground(img, boxes)
            h, w, _ = img.shape
            # put on four background
            four_back_ground[offset_y: offset_y + h, offset_x: offset_x+w] = img
            boxes[:, [0, 2]] += offset_x
            boxes[:, [1, 3]] += offset_y

            boxes_vec.append(
                boxes
            )
            class_vec.append(
                classes
            )

        img = four_back_ground
        boxes = np.concatenate(boxes_vec, axis=0)
        classes = np.concatenate(class_vec, axis=0)

        img, boxes = self.__resize_image(
            img,
            boxes,
            self.image_size
        )
        return img, boxes, classes

    def __get_image_label(
            self,
            index,
    ) -> tuple:
        if self.use_mosaic and np.random.randint(2):
            img, boxes, classes = self.__get_mosaic(index)
            if self.use_mixup and np.random.randint(2):
                img2, boxes2, classes2 = self.__get_mosaic(index)
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                boxes = np.concatenate((boxes, boxes2), 0)
                classes = np.concatenate((classes, classes2), 0)

        else:
            img_id = self.ids[index]
            img, boxes, classes = self.__get_origin_one(img_id)

        return img, boxes, classes

    def __getitem__(self, index):
        img, boxes, classes = self.__get_image_label(index)

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


def get_stronger_voc_data_loader(
        root_path: str,
        years: list,
        kinds_name: list,
        image_size: tuple,
        batch_size: int,
        train: bool = True,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        use_mosaic: bool = True,
        use_mixup: bool = True
):

    if train:
        from Tool.BaseTools.dataaugmentation import ConvertFromInts, PhotometricDistort, RandomMirror, Resize, Normalize

        transform_train = SSDAugmentation(
            size=image_size[0],
            mean=mean,
            std=std,
            augment=[
                ConvertFromInts(),
                PhotometricDistort(),
                RandomMirror(),
                Resize(image_size[0]),
                Normalize(mean, std)
            ]
        )

        train_d = StrongerVOCDataSet(
            root=root_path,
            years=years,
            kinds_name=kinds_name,
            train=True,
            image_size=image_size,
            transform=transform_train,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup
        )

        train_l = DataLoader(train_d,
                             batch_size=batch_size,
                             collate_fn=StrongerVOCDataSet.collate_fn,
                             shuffle=True)
        return train_l
    else:
        transform_test = BaseAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
        )

        test_d = StrongerVOCDataSet(
            root=root_path,
            years=years,
            kinds_name=kinds_name,
            train=False,
            image_size=image_size,
            transform=transform_test,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup
        )

        test_l = DataLoader(test_d,
                            batch_size=batch_size,
                            collate_fn=StrongerVOCDataSet.collate_fn,
                            shuffle=False)
        return test_l
