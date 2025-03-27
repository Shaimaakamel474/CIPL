import os
import numpy as np
import random
from PIL import ImageFilter
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from data.ChestDataset import ChestDataset
from data.ODIRDataset import ODIRDataset
from data.augmentations.rand_augment import RandAugment, Lighting

from settings import img_size
train_h, train_w = img_size, img_size  # 512 (NIH), 640 (ODIR)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[1.0, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GammaAjustment(object):
    """Gamma Correction by one of the given gammas."""

    def __init__(self, gamma=[0.9, 1.1]):   # gamma smaller than 1 make dark regions lighter.
        self.gamma = gamma

    def __call__(self, x):
        gamma = random.uniform(self.gamma[0], self.gamma[1])
        return TF.adjust_gamma(x, gamma)


def get_aug_trans_xray_warmup(n=2, m=9):
    return transforms.Compose([
                RandAugment(n=n, m=m),
                transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.65, 1.0), ratio=(0.65, 1.35)),
                transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.01, 0.01)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-25, 25, -25, 25), fill=0),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def get_aug_trans_xray(n=2, m=9, extend=32):

    trans_init = transforms.Compose([
                  transforms.RandomResizedCrop(size=[train_h + extend, train_w + extend], scale=(0.50, 1.0), ratio=(0.5, 2)),
                  RandAugment(n=n, m=m),
                 ])

    trans_affine = transforms.Compose([      
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=(-40, 40, -40, 40),  fill=0),
                 transforms.RandomVerticalFlip(p=0.5),
                 # transforms.RandomApply([GammaAjustment([0.95, 1.05])], p=0.5),
                 # transforms.RandomPosterize(bits=6, p=0.5),
                 ])

    trans_color = transforms.Compose([
                 transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.01, 0.01)),
                 transforms.RandomApply([GaussianBlur([0.5, 2.0])], p=0.5),
                 # transforms.RandomApply([GammaAjustment([0.95, 1.05])], p=0.5),
                 transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.95, 1.0), ratio=(0.95, 1.05)),
                 ])

    trans_end = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])

    return (trans_init, trans_affine, trans_color, trans_end)
    

def config_dataset_xray(params, warmup=False):
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    torch.cuda.manual_seed(params.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True

    if warmup:
        transform_train = get_aug_trans_xray_warmup(n=1, m=30)
    else:
        transform_train = get_aug_trans_xray(n=1, m=30)

    transform_push = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test_val = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = params.root_dir

    train_dataset = ChestDataset(root_dir=root_dir, transform=transform_train,  mode='train', warmup=warmup)
    train_push_dataset = ChestDataset(root_dir=root_dir, transform=transform_push,  mode='push', warmup=warmup)

    test_dataset = ChestDataset(root_dir=root_dir, transform=transform_test_val,  mode='test', warmup=warmup)   # 25596 test samples
    valid_dataset = ChestDataset(root_dir=root_dir, transform=transform_test_val,  mode='test', warmup=warmup)

    def make_weights_for_balanced_classes(images, nclasses, warmup):
        count = images.sum(axis=0).tolist()
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        # re-balance the training sample according to class occurrence in the training set, the raw weight is:
        # [12.617149758454106,
        #  61.200937316930286,
        #  12.064903568541402,
        #  7.580177042519228,
        #  25.897372335151214,
        #  22.189889549702634,
        #  119.2579908675799,
        #  39.616989002654535,
        #  36.630434782608695,
        #  75.81277213352685,
        #  73.41531974701336,
        #  83.50919264588329,
        #  46.59678858162355,
        #  740.9219858156029,
        #  2.0687128712871288]
        if warmup:
            weight_per_class = [12.617149758454106,
                                61.200937316930286,
                                12.064903568541402,
                                7.580177042519228,
                                25.897372335151214,
                                22.189889549702634,
                                100,
                                39.616989002654535,
                                36.630434782608695,
                                75.81277213352685,
                                73.41531974701336,
                                83.50919264588329,
                                46.59678858162355,
                                150,
                                2.0687128712871288 * 2  # 2
                                ]
        else:
            weight_per_class = [12.617149758454106,
                                61.200937316930286,
                                12.064903568541402,
                                7.580177042519228,
                                25.897372335151214,
                                22.189889549702634,
                                100,
                                39.616989002654535,
                                36.630434782608695,
                                75.81277213352685,
                                73.41531974701336,
                                83.50919264588329,
                                46.59678858162355,
                                150,
                                2.0687128712871288 * 3  # 2
                                ]

        weight_per_class = np.array(weight_per_class)
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = np.mean(weight_per_class[val == 1]).item()
        return weight

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(train_dataset.gr, params.num_classes, warmup)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, shuffle=False, sampler=sampler, drop_last=False, num_workers=0)
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=1)  # 8
    # train_push_loader = DataLoader(test_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=1)  # 8
    test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)

    return train_loader, train_push_loader, test_loader, valid_loader


def get_aug_trans_fundus(n=2, m=9, extend=32):

    trans_init = transforms.Compose([
                  transforms.RandomResizedCrop(size=[train_h + extend, train_w + extend], scale=(0.65, 1.0), ratio=(0.65, 1.35)),
                  RandAugment(n=n, m=m),
                 ])

    trans_affine = transforms.Compose([
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-25, 25, -25, 25),  fill=0),
                 transforms.RandomVerticalFlip(p=0.5),
                 ])

    trans_color = transforms.Compose([
                 transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.01, 0.01)),
                 transforms.RandomApply([GaussianBlur([0.5, 2.0])], p=0.5),
                 # transforms.RandomApply([GammaAjustment([0.95, 1.05])], p=0.5),
                 transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.95, 1.0), ratio=(0.95, 1.05)),
                 ])

    trans_end = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])

    return (trans_init, trans_affine, trans_color, trans_end)


def config_dataset_fundus(params, warmup=False):
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    torch.cuda.manual_seed(params.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True

    transform_train = get_aug_trans_fundus(n=1, m=30)

    transform_push = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test_val = transforms.Compose(
        [
            transforms.Resize([train_h, train_w]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    root_dir = '/mnt/projects/data/OIA-ODIR/'
    train_csv = 'training annotation (English).xlsx'
    test_on_csv = 'on-site test annotation (English).xlsx'
    test_off_csv = 'off-site test annotation (English).xlsx'

    train_dataset = ODIRDataset(label_file=train_csv, root_dir=root_dir, sub_dir='Training Set/Annotation', transform=transform_train, mode='train', warmup=warmup)
    train_push_dataset = ODIRDataset(label_file=train_csv, root_dir=root_dir, sub_dir='Training Set/Annotation', transform=transform_push, warmup=warmup)

    test_dataset_on = ODIRDataset(label_file=test_on_csv, root_dir=root_dir, sub_dir='On-site Test Set/Annotation', transform=transform_test_val, warmup=warmup)  # on site 1972 test samples
    test_dataset_off = ODIRDataset(label_file=test_off_csv, root_dir=root_dir, sub_dir='Off-site Test Set/Annotation', transform=transform_test_val, warmup=warmup)  # off site 991 test samples
    valid_dataset = ODIRDataset(label_file=test_on_csv, root_dir=root_dir, sub_dir='On-site Test Set/Annotation',  transform=transform_test_val, warmup=warmup)

    def make_weights_for_balanced_classes(images, nclasses, warmup):
        count = images.sum(axis=0).tolist()
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        # re-balance the training sample according to class occurrence in the training set
        # [4.1510272071071626,
        #  22.932515337423315,
        #  23.884984025559106,
        #  26.7,
        #  38.73575129533679,
        #  27.895522388059703,
        #  6.23,
        #  2.4155088852988693]
        weight_per_class = [4.1510272071071626,
                            22.932515337423315,
                            23.884984025559106,
                            26.7,
                            38.73575129533679,
                            27.895522388059703,
                            6.23,  # 6.23,
                            2.4155088852988693 * 3  # 3
                            ]

        # weight_per_class = [w if w < 100 else 100 for w in weight_per_class]
        weight_per_class = np.array(weight_per_class)
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = np.mean(weight_per_class[val == 1]).item()
        return weight

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(train_dataset.gr, params.num_classes, warmup)  # 8
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, shuffle=False, sampler=sampler, drop_last=False, num_workers=16)  # 8
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=16)  # 8

    test_loader_on = DataLoader(test_dataset_on, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=16)
    test_loader_off = DataLoader(test_dataset_off, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=16)

    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=16)

    print('Number of train images: {}'.format(len(train_dataset)))
    print('Number of train push images: {}'.format(len(train_push_dataset)))
    print('Number of valid images: {}'.format(len(valid_dataset)))
    print('Number of On test images: {}'.format(len(test_dataset_on)))
    print('Number of Off test images: {}'.format(len(test_dataset_off)))
    print('')

    return train_loader, train_push_loader, test_loader_on, test_loader_off, valid_loader






