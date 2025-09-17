import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as transF
from torchvision import transforms
import matplotlib.pyplot as plt
import random


Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}

mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


from settings import img_size
train_h, train_w = img_size, img_size   # 512

class ChestDataset(Dataset):
    def __init__(self, root_dir, transform, mode, warmup) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.warmup = warmup

        # Paths ثابتة
        txt_csv_dir = "/kaggle/input/nih-total"
        images_dir = "/kaggle/input/tst-shaima"

        # تحديد ملف القوائم
        label_file = 'train_val_list.txt' if self.mode in ['train', 'push'] else 'test_list.txt'

        # قراءة CSV بشكل صحيح
        gr_path = os.path.join(txt_csv_dir, "Data_Entry_2017.csv")
        gr_df = pd.read_csv(gr_path)
        gr = dict(zip(gr_df["Image Index"], gr_df["Finding Labels"]))

        # قراءة TXT
        img_list = os.path.join(txt_csv_dir, label_file)
        with open(img_list) as f:
            all_names = f.read().splitlines()

        self.all_imgs = np.asarray(all_names)
        self.gr_str = np.asarray([gr[i] for i in self.all_imgs])
        self.gr = np.zeros((self.gr_str.shape[0], len(Labels)))
        for idx, i in enumerate(self.gr_str):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[t] for t in target]]).squeeze()
            self.gr[idx] = binary_result

        self.same_class_index = [np.where(self.gr[:, i] == 1)[0] for i in range(self.gr.shape[1])]
        self.images_dir = images_dir  # تخزين مسار الصور

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        image_path1 = os.path.join(self.images_dir, self.all_imgs[index])
        image1 = Image.open(image_path1).convert("RGB")
        label1 = self.gr[index]

        if self.mode == 'train' and not self.warmup:
            posi_index = random.choice(np.where(label1 == 1)[0])
            index2 = random.choice(self.same_class_index[posi_index])
            image_path2 = os.path.join(self.images_dir, self.all_imgs[index2])
            image2 = Image.open(image_path2).convert("RGB")
            label2 = self.gr[index2]

            transform_common = transforms.Compose([self.transform[0], self.transform[1]])
            transform_specific = transforms.Compose([self.transform[2], self.transform[3]])

            image1_common = transform_common(image1)
            data1 = transform_specific(image1_common)
            data1_aug = transform_specific(image1_common)

            image2_common = transform_common(image2)
            data2 = transform_specific(image2_common)
            data2_aug = transform_specific(image2_common)

            if random.random() > 0.5:
                data1, data1_aug = transF.hflip(data1), transF.hflip(data1_aug)
            if random.random() > 0.5:
                data2, data2_aug = transF.hflip(data2), transF.hflip(data2_aug)
            if random.random() > 0.5:
                data1, data1_aug = transF.vflip(data1), transF.vflip(data1_aug)
            if random.random() > 0.5:
                data2, data2_aug = transF.vflip(data2), transF.vflip(data2_aug)

            return data1, data2, torch.tensor(label1).long(), torch.tensor(label2).long(), data1_aug, data2_aug
        else:
            data = self.transform(image1)
            target = torch.tensor(label1).long()
            return data, target, image_path1

            # return data, target, self.gr_str[index]

