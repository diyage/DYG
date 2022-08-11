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
        self.img_size = self.image_size[0]
        self.kinds_name = kinds_name

    def __get_one_origin(
            self,
            index
    ):
        root_path, xml_file_name = self.image_and_xml_path_info[index]
        xml_trans = XMLTranslate(root_path=root_path, file_name=xml_file_name)
        # xml_trans.resize(new_size=self.image_size)

        img, label = xml_trans.img, xml_trans.objects
        new_label = []
        for i in range(len(label)):
            kind_index = self.kinds_name.index(label[i][0])
            new_label.append(
                (*label[i][1:], kind_index)
            )

        w, h = xml_trans.img_size[0], xml_trans.img_size[1]
        targets = np.array(new_label).astype(np.float32)
        return img, targets, h, w

    def load_mosaic(self, index):
        ids_list_ = self.ids[:index] + self.ids[index + 1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        for id_ in ids:
            img_i, target_i, _, _ = self.__get_one_origin(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        mean = np.array([v * 255 for v in self.transform.mean])
        mosaic_img = np.ones([self.img_size * 2, self.img_size * 2, 3], dtype=np.uint8) * mean
        # mosaic center
        yc, xc = [
            int(np.random.uniform(-x, 2 * self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]
        ]
        # yc = xc = self.img_size

        mosaic_tg = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            target_i = np.array(target_i)
            h0, w0, _ = img_i.shape

            # resize
            scale_range = np.arange(50, 210, 10)
            s = np.random.choice(scale_range) / 100.

            if np.random.randint(2):
                # keep aspect ratio
                r = self.img_size / max(h0, w0)
                if r != 1:
                    img_i = CV2.resize(img_i, (int(w0 * r * s), int(h0 * r * s)))
            else:
                img_i = CV2.resize(img_i, (int(self.img_size * s), int(self.img_size * s)))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            target_i_ = target_i.copy()
            if len(target_i) > 0:
                # a valid target, and modify it.
                target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                target_i_[:, 3] = (h * (target_i[:, 3]) + padh)
                # check boxes
                valid_tgt = []
                for tgt in target_i_:
                    x1, y1, x2, y2, label = tgt
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 5. and bh > 5.:
                        valid_tgt.append([x1, y1, x2, y2, label])
                if len(valid_tgt) == 0:
                    valid_tgt.append([0., 0., 0., 0., 0.])

                mosaic_tg.append(target_i_)
        # check target
        if len(mosaic_tg) == 0:
            mosaic_tg = np.zeros([1, 5])
        else:
            mosaic_tg = np.concatenate(mosaic_tg, axis=0)
            # Cutout/Clip targets
            np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
            # normalize
            mosaic_tg[:, :4] /= (self.img_size * 2)

        return mosaic_img, mosaic_tg, self.img_size, self.img_size

    def __get_image_label(
            self,
            index,
    ) -> tuple:
        if self.use_mosaic and np.random.randint(2):
            img, target, height, width = self.load_mosaic(index)
            if self.use_mixup and np.random.randint(2):
                img2, target2, height, width = self.load_mosaic(np.random.randint(0, len(self.ids)))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                target = np.concatenate((target, target2), 0)

            boxes = target[:, :4]
            classes = target[:, 4]

        else:
            img_id = self.ids[index]
            img, target, height, width = self.__get_one_origin(img_id)
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            # augment
            boxes = target[:, :4]
            classes = target[:, 4]

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
        transform_train = SSDAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
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
