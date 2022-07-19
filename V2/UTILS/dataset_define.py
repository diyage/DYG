from Tool import VOC2012DataSet, get_imagenet_dataset
from V2.UTILS.config_define import DataSetConfig, TrainConfig
from torch.utils.data import DataLoader
from torchvision import transforms


def get_voc_data_loader(
        data_opt: DataSetConfig,
        train_opt: TrainConfig,

):
    normalize = transforms.Normalize(
            std=[0.5, 0.5, 0.5],
            mean=[0.5, 0.5, 0.5],
        )
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transform_train

    train_d = VOC2012DataSet(root=data_opt.root_path,
                             train=True,
                             image_size=data_opt.image_size,
                             transform=transform_train)

    train_l = DataLoader(train_d,
                         batch_size=train_opt.batch_size,
                         collate_fn=VOC2012DataSet.collate_fn,
                         shuffle=True)

    test_d = VOC2012DataSet(root=data_opt.root_path,
                            train=False,
                            image_size=data_opt.image_size,
                            transform=transform_test)

    test_l = DataLoader(test_d,
                        batch_size=train_opt.batch_size,
                        collate_fn=VOC2012DataSet.collate_fn,
                        shuffle=False)
    return train_l, test_l


def get_image_net_224_loader(
        data_opt: DataSetConfig,
        train_opt: TrainConfig,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_set = get_imagenet_dataset(
        root=data_opt.image_net_dir,
        transform=transform_train,
        train=True
    )
    test_set = get_imagenet_dataset(
        root=data_opt.image_net_dir,
        transform=transform_test,
        train=False
    )
    train_loader = DataLoader(
        train_set,
        batch_size=train_opt.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=train_opt.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def get_image_net_448_loader(
        data_opt: DataSetConfig,
        train_opt: TrainConfig,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,

    ])

    transform_test = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = get_imagenet_dataset(
        root=data_opt.image_net_dir,
        transform=transform_train,
        train=True
    )
    test_set = get_imagenet_dataset(
        root=data_opt.image_net_dir,
        transform=transform_test,
        train=False
    )
    train_loader = DataLoader(
        train_set,
        batch_size=train_opt.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=train_opt.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
