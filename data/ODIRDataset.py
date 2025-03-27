import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as transF
from torchvision import transforms
import matplotlib.pyplot as plt
import random


Labels = {
    "Normal": 7,
    "Diabetic Retinopathy": 0,
    "Glaucoma": 1,
    "Cataract": 2,
    "AMD": 3,
    "Hypertensive Retinopathy": 4,
    "Myopia": 5,
    "Other Diseases": 6
}

mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7])

from settings import img_size
train_h, train_w = img_size, img_size   # 640


deterministic = [
        ['suspected diabetic retinopathy', 'suspected moderate nonproliferative retinopathy', 'suspicious diabetic retinopathy', 'suspected moderate non proliferative retinopathy', "mild nonproliferative retinopathy", "mild non proliferative retinopathy", 'moderate nonproliferative retinopathy', 'moderate non proliferative retinopathy', 'severe nonproliferative retinopathy', 'severe non proliferative retinopathy', 'proliferative diabetic retinopathy', 'severe proliferative diabetic retinopathy', 'diabetic retinopathy'],
        ["glaucoma", 'suspected glaucoma'],
        ["cataract", 'suspected cataract'],
        ["age-related macular degeneration", "dry age-related macular degeneration", "wet age-related macular degeneration"],
        ['mild hypertension', 'sever hypertension', "hypertensive retinopathy"],
        ['suspected pathological myopia', "myopia retinopathy", 'pathological myopia', 'myopic retinopathy', 'myopic maculopathy'],
]


class ODIRDataset(Dataset):
    def __init__(self, label_file, root_dir, sub_dir, transform, mode='test', warmup=False) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.sub_dir = sub_dir
        self.mode = mode
        self.warmup = warmup

        df = pd.read_excel(os.path.join(root_dir, sub_dir, label_file))

        Index = np.asarray(df['ID'])

        all_Diagnostic_left = np.asarray(df['Left-Diagnostic Keywords'])
        all_Diagnostic_right = np.asarray(df['Right-Diagnostic Keywords'])

        all_imgs_left = np.asarray(df['Left-Fundus'])
        all_imgs_right = np.asarray(df['Right-Fundus'])

        gr_ori = np.array(df[['D', 'G', 'C', 'A', 'H', 'M', 'O', 'N']])   # normal is the last class

        ################################################################################################
        gr_left = np.zeros_like(gr_ori)
        nnn = 0
        for i in range(len(gr_ori)):
            keyword_left = all_Diagnostic_left[i]

            if keyword_left == 'anterior segment image' or keyword_left == 'no fundus image' or keyword_left == 'low image quality' or keyword_left == 'optic disk photographically invisible' or keyword_left == 'lens dust' or keyword_left == 'image offset':
                continue

            # if 'normal fundus' in keyword_left:
            #     gr_left[i, 7] = 1

            for key in keyword_left.split('，'):
                if key == 'lens dust': continue
                if key == 'optic disk photographically invisible': continue
                if key == 'low image quality': continue
                if key == 'no fundus image': continue
                if key == 'image offset': continue
                if key == 'anterior segment image': continue
                if key in deterministic[0]:
                    gr_left[i, 0] = 1
                elif key in deterministic[1]:
                    gr_left[i, 1] = 1
                elif key in deterministic[2]:
                    gr_left[i, 2] = 1
                elif key in deterministic[3]:
                    gr_left[i, 3] = 1
                elif key in deterministic[4]:
                    gr_left[i, 4] = 1
                elif key in deterministic[5]:
                    gr_left[i, 5] = 1
                elif key == 'normal fundus':
                    gr_left[i, 7] = 1
                else:
                    nnn = nnn + 1
                    gr_left[i, 6] = 1
                    # print(Index[i], all_imgs_left[i], keyword_left)
                    # print(Index[i], key)
        # correct some 'normal fundus' image
        for i in range(len(gr_left)):
            if (gr_left[i, 0:-1].sum() > 0):
                gr_left[i, -1] = 0
        ################################################################################################
        ################################################################################################
        gr_right = np.zeros_like(gr_ori)
        mmm = 0
        for i in range(len(gr_ori)):
            keyword_right = all_Diagnostic_right[i]

            if keyword_right == 'anterior segment image' or keyword_right == 'no fundus image' or keyword_right == 'low image quality' or keyword_right == 'optic disk photographically invisible' or keyword_right == 'lens dust' or keyword_right == 'image offset':
                continue

            # if 'normal fundus' in keyword_right:
            #     gr_right[i, 7] = 1
            #     continue

            for key in keyword_right.split('，'):
                if key == 'lens dust': continue
                if key == 'optic disk photographically invisible': continue
                if key == 'low image quality': continue
                if key == 'no fundus image': continue
                if key == 'image offset': continue
                if key == 'anterior segment image': continue
                if key in deterministic[0]:
                    gr_right[i, 0] = 1
                elif key in deterministic[1]:
                    gr_right[i, 1] = 1
                elif key in deterministic[2]:
                    gr_right[i, 2] = 1
                elif key in deterministic[3]:
                    gr_right[i, 3] = 1
                elif key in deterministic[4]:
                    gr_right[i, 4] = 1
                elif key in deterministic[5]:
                    gr_right[i, 5] = 1
                elif key == 'normal fundus':
                    gr_right[i, 7] = 1
                else:
                    mmm = mmm + 1
                    gr_right[i, 6] = 1
                    # print(Index[i], all_imgs_right[i], keyword_right)
                    # print(Index[i], key)
        # correct some 'normal fundus' image
        for i in range(len(gr_right)):
            if (gr_right[i, 0:-1].sum() > 0):
                gr_right[i, -1] = 0
        # ################################################################################################
        # print('')
        # print('check with original multi-label ground truth...')
        # aaa = 0
        # for i in range(len(gr_ori)):
        #     gt_comb = gr_left[i] + gr_right[i]
        #     gt_comb[gt_comb == 2] = 1
        #     if gt_comb[0:-1].sum() > 0: gt_comb[-1] = 0
        #     if (gt_comb != gr_ori[i]).any():
        #         aaa = aaa + 1
        #         print(Index[i], all_Diagnostic_left[i], gr_left[i], all_Diagnostic_right[i], gr_right[i], '%%%%%%', gr_ori[i])
        # ################################################################################################

        all_imgs_left = all_imgs_left[gr_left.sum(1) > 0]
        all_imgs_right = all_imgs_right[gr_right.sum(1) > 0]
        gr_left = gr_left[gr_left.sum(1) > 0]
        gr_right = gr_right[gr_right.sum(1) > 0]
        self.all_imgs = np.concatenate([all_imgs_left, all_imgs_right])
        self.gr = np.concatenate([gr_left, gr_right])

        same_class_index = []
        for i in range(self.gr.shape[1]):
            same_class_index.append(np.where(self.gr[:, i] == 1)[0])
        self.same_class_index = same_class_index              # [1801,  326,  313,  280,  193,  268,  1200,  3095]

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        if self.mode == 'train' and self.warmup == False:
            image_path = os.path.join(self.root_dir, self.sub_dir.replace('Annotation', 'Images_crop'), self.all_imgs[index])
            image1 = Image.open(image_path).convert("RGB")
            label1 = self.gr[index]

            posi_index = random.choice(np.where(label1 == 1)[0])
            index2 = random.choice(self.same_class_index[posi_index])
            image2_name = os.path.join(self.root_dir, self.sub_dir.replace('Annotation', 'Images_crop'), self.all_imgs[index2])
            image2 = Image.open(image2_name).convert('RGB')
            label2 = self.gr[index2]

            transform_common = transforms.Compose([self.transform[0], self.transform[1], ])
            transform_specific = transforms.Compose([self.transform[2], self.transform[3], ])

            image1_common = transform_common(image1)
            data1 = transform_specific(image1_common)
            data1_aug = transform_specific(image1_common)

            image2_common = transform_common(image2)
            data2 = transform_specific(image2_common)
            data2_aug = transform_specific(image2_common)

            if random.random() > 0.5:
                data1 = transF.hflip(data1)
                data1_aug = transF.hflip(data1_aug)

            if random.random() > 0.5:
                data2 = transF.hflip(data2)
                data2_aug = transF.hflip(data2_aug)

            if random.random() > 0.5:
                data1 = transF.vflip(data1)
                data1_aug = transF.vflip(data1_aug)

            if random.random() > 0.5:
                data2 = transF.vflip(data2)
                data2_aug = transF.vflip(data2_aug)

            target1 = torch.tensor(label1).long()
            target2 = torch.tensor(label2).long()

            return data1, data2, target1, target2, data1_aug, data2_aug
        else:
            img_path = os.path.join(self.root_dir, self.sub_dir.replace('Annotation', 'Images_crop'), self.all_imgs[index])
            img = Image.open(img_path).convert("RGB")
            data = self.transform(img)
            target = torch.tensor(self.gr[index]).long()
            return data, target, img_path
            # return data, target, self.gr_str[index]
