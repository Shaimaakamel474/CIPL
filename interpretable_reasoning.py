import os
import torch
import torch.utils.data
import re
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from helpers import makedir
import model
from utlis.utlis_func import *
from matplotlib import cm
import pandas as pd


Labels_dict = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltrate": 3,
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


def distance_2_similarity_exp(distances):
    return torch.exp(-distances / 256.0)  # 128.0


def ind2name(jjj):
    save_name = str(jjj + 1)
    if jjj < 9:
        save_name = '0' + save_name
    else:
        save_name = save_name
    return save_name


def get_one_heatmap(similarity_map, gt_box, img_np):

    original_img_size2, original_img_size1 = img_np.shape[0], img_np.shape[1]

    proto_act_img_j = similarity_map.squeeze().detach().cpu().numpy()

    upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size2, original_img_size1), interpolation=cv2.INTER_CUBIC)
    rescaled_act_img_j = (upsampled_act_img_j - np.amin(upsampled_act_img_j)) / (np.amax(upsampled_act_img_j) - np.amin(upsampled_act_img_j))

    proto_bound_j = gt_box

    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    original_img_j = np.array(img_np).astype(float) / 1.0
    overlayed_original_img_j = 0.7 * original_img_j + 0.3 * heatmap
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * overlayed_original_img_j), cv2.COLOR_RGB2BGR)

    cv2.rectangle(img_bgr_uint8, (proto_bound_j[2], proto_bound_j[0]), (proto_bound_j[3] - 1, proto_bound_j[1] - 1), color=(100, 255, 100), thickness=2)

    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255

    return img_rgb_float, rescaled_act_img_j


from settings import base_architecture, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, root_dir

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


# construct the model
ppnet = model.construct_CIPL(base_architecture=base_architecture,
                             pretrained=True, img_size=img_size,
                             prototype_shape=prototype_shape,
                             num_classes=num_classes,
                             prototype_activation_function=prototype_activation_function,
                             add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()

checkpoint_path = "14nopush0.8232.pth"

ppnet.load_state_dict(torch.load(checkpoint_path))
model = torch.nn.DataParallel(ppnet)
model.eval()

for p in model.module.parameters():
    p.requires_grad = False
connection_weight_allclass = model.module.last_layer.weight.squeeze()

transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
transform_ori = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            ])

gr_path = os.path.join(root_dir, "BBox_List_2017.csv")
df_box = pd.read_csv(gr_path)

save_dir_main = './output_vis'
makedir(save_dir_main)


image_name = '00028027_000.png'
target_name = "Mass"


if __name__ == '__main__':

    img_idx = df_box.loc[df_box['Image Index'] == image_name].index
    box_x = int(df_box.iloc[img_idx]['Bbox [x'] / 2.0)
    box_y = int(df_box.iloc[img_idx]['y'] / 2.0)
    box_w = int(df_box.iloc[img_idx]['w'] / 2.0)
    box_h = int(df_box.iloc[img_idx]['h]'] / 2.0)
    gt_box = (box_y, box_y + box_h, box_x, box_x + box_w)
    label_gt = Labels_dict[target_name]

    img_PIL = Image.open(os.path.join(root_dir, 'data', image_name)).convert('RGB')
    img_input = transform(img_PIL)

    with torch.no_grad():
        img_input = img_input.unsqueeze(0).cuda()
        target = torch.tensor(label_gt).cuda()

        output_all, min_distances_all, similarities_all = model.module.forward_infer(img_input)

        # compute predictions
        output_fg = output_all[:, 0:num_classes-1]
        output_bg = output_all[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last group of prototypes
        output_new = torch.stack((output_bg, output_fg), dim=-1)
        prob = torch.softmax(output_new.squeeze(0), dim=1)[label_gt][1]
        print(image_name, 'prob:', round(prob.item(), 3))

        img_np = np.array(transform_ori(img_PIL)).transpose(1, 2, 0).astype(np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        original_img_size1, original_img_size2 = img_np.shape[0], img_np.shape[1]
        num_proto_per_class = model.module.num_prototypes // model.module.num_classes  # 50
        proto_index_start = num_proto_per_class * label_gt
        proto_index_end = num_proto_per_class * label_gt + num_proto_per_class

        similarity_maps = similarities_all[0, proto_index_start:proto_index_end]
        min_distances = min_distances_all[0, proto_index_start:proto_index_end]
        connection_weight = connection_weight_allclass[label_gt, proto_index_start:proto_index_end]

        # _, sorted_index = torch.sort(min_distances * connection_weight, descending=False)
        _, sorted_index = torch.sort(min_distances, descending=False)
        similarity = distance_2_similarity_exp(min_distances)

        for iii, index in enumerate(sorted_index.cpu().numpy()):
            score = np.around(similarity[index].cpu().numpy(), decimals=3)
            weight = np.around(connection_weight[index].cpu().numpy(), decimals=3)
            overlap_map, heatmap = get_one_heatmap(similarity_maps[index], gt_box, img_np)
            save_dir = os.path.join(save_dir_main, target_name)
            makedir(save_dir)
            save_path = os.path.join(save_dir, image_name.replace('.png', '_' + ind2name(iii) + '_overlap.jpg'))
            plt.imsave(save_path, overlap_map, vmin=0.0, vmax=1.0, pil_kwargs={'quality': 99})

        save_path = os.path.join(save_dir, image_name.replace('.png', '_ori.png'))
        plt.imsave(save_path, img_np, vmin=0.0, vmax=1.0, cmap=cm.gray, pil_kwargs={'quality': 99})

