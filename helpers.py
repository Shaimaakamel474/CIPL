import os
import torch
import numpy as np
import cv2

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def list_of_similarities_2d(X, Y):
    return - torch.sum((X.unsqueeze(dim=3) - Y.unsqueeze(dim=2)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

# def find_high_activation_crop(activation_map, percentile=95):
#     threshold = np.percentile(activation_map, percentile)
#     mask = np.ones(activation_map.shape)
#     mask[activation_map < threshold] = 0
#     lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
#     for i in range(mask.shape[0]):
#         if np.amax(mask[i]) > 0.5:
#             lower_y = i
#             break
#     for i in reversed(range(mask.shape[0])):
#         if np.amax(mask[i]) > 0.5:
#             upper_y = i
#             break
#     for j in range(mask.shape[1]):
#         if np.amax(mask[:,j]) > 0.5:
#             lower_x = j
#             break
#     for j in reversed(range(mask.shape[1])):
#         if np.amax(mask[:,j]) > 0.5:
#             upper_x = j
#             break
#     return lower_y, upper_y+1, lower_x, upper_x+1


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0

    # from PIL import Image
    # image_output = Image.fromarray(mask * 255)
    # image_output.convert('RGB').save("image_output.jpg")

    highest_index = list(np.unravel_index(np.argmax(activation_map), activation_map.shape))

    n_labels, img_labeled, lab_stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, n_labels):
        temp = img_labeled == label
        if temp[highest_index[0], highest_index[1]] == False:
            mask[temp] = 0

    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break

    if upper_x <= lower_x:
        upper_x = lower_x

    if upper_y <= lower_y:
        upper_y = lower_y

    return (lower_y, upper_y + 1, lower_x, upper_x + 1), highest_index


from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr