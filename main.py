import os
import torch
import time
import argparse
import re
import wandb
from helpers import makedir
import model
import push
import train_and_test as tnt
import save
from .log import create_logger

from utlis.utlis_func import *


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')     # "0, 1"
parser.add_argument('-seed', type=int, default=42)
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])

os.environ['WANDB_START_METHOD'] = 'fork'
os.environ["WANDB__SERVICE_WAIT"] = "1500"


# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, coefs, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


model_dir = 'saved_models/{}/'.format(datestr()) + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)


log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import root_dir, train_batch_size, test_batch_size, train_push_batch_size


args.train_batch_size = train_batch_size
args.test_batch_size = test_batch_size
args.train_push_batch_size = train_push_batch_size
args.coefs = coefs
args.num_classes = num_classes
args.img_size = img_size
args.root_dir = root_dir
args.model_dir = model_dir


# WandB – Initialize a new run
wandb.init(project='CIPL', mode='disabled')     # mode='disabled'
wandb.run.name = wandb.run.id + '_nih'


train_loader_warmup, train_push_loader_warmup, test_loader_warmup, valid_loader_warmup = config_dataset_xray(args, warmup=True)
train_loader, train_push_loader, test_loader, valid_loader = config_dataset_xray(args)


# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_CIPL(base_architecture=base_architecture,
                             pretrained=True, img_size=img_size,
                             prototype_shape=prototype_shape,
                             num_classes=num_classes,
                             prototype_activation_function=prototype_activation_function,
                             add_on_layers_type=add_on_layers_type)


# ppnet.load_state_dict(torch.load("31nopush0.8275.pth"))


ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

weight_decay = 0e-3

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': weight_decay},
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs, )
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=1, gamma=0.5)   # 0.5


from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': weight_decay},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
for epoch in range(0, num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < 0:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader_warmup, optimizer=warm_optimizer, warmup=True,
                      class_specific=class_specific, coefs=coefs, log=log)
    elif epoch < num_warm_epochs:
        tnt.joint(model=ppnet_multi, log=log)
        if epoch in [10, ]:
            joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader_warmup, optimizer=joint_optimizer, warmup=True,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        if epoch in [20, 25, 30, 35]:
            joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, warmup=False,
                      class_specific=class_specific, coefs=coefs, log=log)

    log('###### lr: \t{0}'.format(joint_optimizer.param_groups[0]['lr']))

    wandb.log({
        "LR": joint_optimizer.param_groups[0]['lr'],
        "Epoch": epoch,
    })
    auc_nih = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', auc=auc_nih,
                                target_auc=0.60, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader,
            prototype_network_parallel=ppnet_multi,
            class_specific=class_specific,
            preprocess_input_function=None,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        auc_nih = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', auc=auc_nih,
                                    target_auc=0.60, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(6):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, train_last=True,
                              class_specific=class_specific, coefs=coefs, log=log)
                auc_nih = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push',
                                            auc=auc_nih, target_auc=0.60, log=log)
   
logclose()

