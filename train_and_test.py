import time
import torch
import torch.nn.functional as F
from helpers import list_of_distances, make_one_hot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb


def cluster_sep_loss_fn(model, min_distances, label):
    max_dist = (model.module.prototype_shape[1]
                * model.module.prototype_shape[2]
                * model.module.prototype_shape[3]) ** 2  ################
    batch_size = label.shape[0]
    cluster_cost = 0.0
    separation_cost = 0.0
    for b in range(batch_size):
        real_labels = torch.where(label[b] == 1)[0]
        multiple_cluster = []
        for one_label in real_labels:
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, one_label]).cuda()
            inverted_distances = torch.max((max_dist - min_distances[b]) * prototypes_of_correct_class)
            multiple_cluster.append(max_dist - inverted_distances)
        cluster_cost += sum(multiple_cluster)
        # cluster_cost += sum(multiple_cluster) / len(multiple_cluster)

        prototypes_of_wrong_class = 1 - torch.t(model.module.prototype_class_identity[:, real_labels]).cuda()
        prototypes_of_wrong_class = prototypes_of_wrong_class.all(dim=0) * 1.0
        inverted_distances_to_nontarget_prototypes = torch.max((max_dist - min_distances[b]) * prototypes_of_wrong_class)
        separation_cost += max_dist - inverted_distances_to_nontarget_prototypes

    return cluster_cost / batch_size, separation_cost / batch_size


def l1_loss_fn(model):
    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
    loss_l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    return loss_l1


def _binary_cross_entropy(h, t):
    return -t*torch.log(h)-(1-t)*torch.log(1-h)


def normalize_l2(x):
    return x / x.norm(2, dim=1, keepdim=True)


def dist2simi(x):   # [0, 1]
    return 1 - x


def multilabel_ce_loss(output, target, num_classes, ignore_nofinding=False):
    if ignore_nofinding == False:
        output_fg = output[:, 0:num_classes-1]
        output_bg = output[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
        output_new = torch.stack((output_bg, output_fg), dim=-1).permute(0, 2, 1)  # [B, 2, 14]
        cross_entropy = F.cross_entropy(output_new, target[:, 0:num_classes-1])
        return cross_entropy
    else:
        output_fg = output[:, 0:num_classes-1]
        output_bg = output[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
        output_new = torch.stack((output_bg, output_fg), dim=-1).permute(0, 2, 1)  # [B, 2, 14]
        cross_entropy_total = torch.tensor(0.0).cuda()
        n_samples = 0.0
        for b in range(output_new.shape[0]):
            if target[b, 0:num_classes-1].sum() == 0:
                continue
            else:
                cross_entropy_total += F.cross_entropy(output_new[b].unsqueeze(0), target[b, 0:num_classes-1].unsqueeze(0))
                n_samples += 1
        return cross_entropy_total / (n_samples + 1e-7)


def align_gram_batch(student, teacher, num_classes, temperature=1.0):  # 2.0 (softer)
    student_fg = student[:, 0:num_classes-1]
    student_bg = student[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    student_new = torch.stack((student_bg, student_fg), dim=-1)    # [b, 7, 2]

    teacher_fg = teacher[:, 0:num_classes-1]
    teacher_bg = teacher[:, -1].unsqueeze(1).repeat(1, num_classes-1)  # no finding is the last prototypes
    teacher_new = torch.stack((teacher_bg, teacher_fg), dim=-1)    # [b, 7, 2]

    loss_kd = bc_loss(student_new, teacher_new, temperature)

    return loss_kd


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num, _ = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=2)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=2)

    student_matrix = torch.bmm(pred_student.permute(1, 0, 2), pred_student.permute(1, 2, 0))   # (C-1) * B * B
    teacher_matrix = torch.bmm(pred_teacher.permute(1, 0, 2), pred_teacher.permute(1, 2, 0))   # (C-1) * B * B

    if reduce:
        # consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / (class_num * batch_size * batch_size)
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss
    

def align_loss_cos(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = (F.normalize(inputs, dim=1) * F.normalize(targets, dim=1)).sum(1)    # number of prototype
    loss = (1 - loss).sum([1, 2])
    return loss.mean()


def align_loss_l2(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = F.mse_loss(inputs, targets, reduction='none').sum(1)    # mean
    loss = loss.mean([1, 2])
    return loss.mean()


def _training_warmup(model, dataloader, optimizer=None, train_last=False, class_specific=True, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    predictions = []
    all_targets = []
    for i, (image, label, _) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad()
        with grad_req:

            output, min_distances, _ = model.module.forward_infer(input)
            output_fg = output[:, 0:model.module.num_classes-1]
            output_bg = output[:, -1].unsqueeze(1).repeat(1, model.module.num_classes-1)  # no finding is the last group of prototypes
            output_new = torch.stack((output_bg, output_fg), dim=-1)
            cross_entropy = F.cross_entropy(output_new.permute(0, 2, 1), target[:, 0:model.module.num_classes-1])

            if class_specific:
                cluster_cost, separation_cost = cluster_sep_loss_fn(model, min_distances, label)
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                separation_cost = torch.mean(min_distance)

            # evaluation statistics
            predicted = torch.argmax(output_new.data, dim=-1)
            n_examples += target.shape[0] * target[:, 0:-1].shape[1]
            n_correct += (predicted == target[:, 0:-1]).sum().item()

            predictions.append(F.softmax(output_new.data, dim=2)[:, :, 1].cpu().numpy())
            all_targets.append(label.numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

            # compute gradient and do SGD step
            loss = (
                    coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * F.relu(2.0 - separation_cost)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clip FC weights to ensure positive connections
            #####################################################################
            if train_last:
                prototype_class_identity = model.module.prototype_class_identity.t()
                weight = model.module.last_layer.weight.data
                weight[prototype_class_identity == 0] = 0  # set negative weight to be 0
                weight = torch.clamp(weight, min=0.0)  # set positive weight to be more than 0
                model.module.last_layer.weight.data = weight
            #####################################################################

            if i % 50 == 0:
                print(
                    '{} {} \tLoss_total: {:.4f} \tLoss_CE: {:.4f} \tLoss_clust: {:.4f} \tLoss_sepa: {:.4f} '
                    '\tAcc: {:.1f}'.format(i, len(dataloader), loss.item(), cross_entropy.item(),
                        cluster_cost.item(), separation_cost.item(),
                        n_correct / (n_examples + 0.000001) * 100))
        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    all_targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    all_auc = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions[:, i]) for i in range(model.module.num_classes - 1)],
    )
    mean_auc = all_auc.mean()

    print('##############TRAIN################')
    print('AUC:', np.around(all_auc, 4) * 100)
    print('Mean AUC:', mean_auc * 100)
    print('##############TRAIN################')

    log('\ttime: \t{0}'.format(end - start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tAUC: \t\t{0}%'.format(np.around(all_auc, 4) * 100))
    log('\tMean AUC: \t\t{0}%'.format(mean_auc * 100))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))

    return mean_auc


def _training(model, dataloader, optimizer=None, train_last=False, class_specific=True, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    predictions = []
    all_targets = []
    for i, (image1, image2, label1, label2, image1_aug, image2_aug) in enumerate(dataloader):
        image1 = image1.cuda()
        image2 = image2.cuda()
        img_comb = [image1, image2]

        image1_aug = image1_aug.cuda()
        image2_aug = image2_aug.cuda()
        img_aug_comb = [image1_aug, image2_aug]

        target = torch.cat([label1, label2])

        label1 = label1.cuda()
        label2 = label2.cuda()
        label = torch.cat([label1, label2])

        label_new = label1 + label2
        label_new[label_new != 2] = 0
        label_new[label_new == 2] = 1
        label_new = torch.cat([label_new, label_new])

        grad_req = torch.enable_grad()
        with grad_req:

            logits, min_distances, co_logits, similarities, simi_ema, logits_ema = model(img_comb, img_aug_comb)

            # compute loss
            cross_entropy = multilabel_ce_loss(logits, label, model.module.num_classes-1)
            cross_coatten_cost = multilabel_ce_loss(co_logits, label_new, model.module.num_classes-1, ignore_nofinding=False)

            pred_align_cost = align_gram_batch(student=logits, teacher=logits_ema.detach(), num_classes=model.module.num_classes-1)

            # inte_align_cost = align_loss_l2(similarities, simi_ema)
            inte_align_cost = align_loss_cos(similarities, simi_ema)

            if class_specific:
                cluster_cost, separation_cost = cluster_sep_loss_fn(model, min_distances, target)
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                separation_cost = torch.mean(min_distance)

            output_fg = logits[:, 0:model.module.num_classes]
            output_bg = logits[:, -1].unsqueeze(1).repeat(1, model.module.num_classes)  # no finding is the last group of prototypes
            output_new = torch.stack((output_bg, output_fg), dim=-1)

            # evaluation statistics
            predicted = torch.argmax(output_new.data, dim=-1)
            n_examples += label.shape[0] * label[:, 0:-1].shape[1]
            n_correct += (predicted == label[:, 0:-1]).sum().item()

            predictions.append(F.softmax(output_new.data, dim=2)[:, :, 1].cpu().numpy())
            all_targets.append(target.numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                loss = (
                        coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * F.relu(2.0 - separation_cost)
                        + coefs['cross_att'] * cross_coatten_cost
                        + coefs['inte_align'] * inte_align_cost
                        + coefs['pred_align'] * pred_align_cost
                        )
            else:
                loss = (coefs['crs_ent'] * cross_entropy + coefs['clst'] * cluster_cost)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # clip FC weights to ensure positive connections
            #####################################################################
            if train_last:
                prototype_class_identity = model.module.prototype_class_identity.t()
                weight = model.module.last_layer.weight.data
                weight[prototype_class_identity == 0] = 0  # set negative weight to be 0
                weight = torch.clamp(weight, min=0.0)  # set positive weight to be more than 0
                model.module.last_layer.weight.data = weight
            #####################################################################

            if i % 50 == 0:    # 20
                print(
                    '{} {} \tLoss_total: {:.4f} \tLoss_CE: {:.4f} \tLoss_clust: {:.4f} \tLoss_sepa: {:.4f}'
                    '\t Loss_cross_atten: {:.4f} \t Loss_inte_align: {:.14f} \t Loss_pred_align: {:.4f} \tAcc: {:.1f}'.format(
                        i, len(dataloader), loss.item(), cross_entropy.item(),
                        cluster_cost.item(), separation_cost.item(), cross_coatten_cost.item(),
                        inte_align_cost.item(), pred_align_cost.item(),
                        n_correct / (n_examples + 0.000001) * 100))

                wandb.log({
                    "Train Total Loss": loss.item(),
                    "Train CE Loss": cross_entropy.item(),
                    "Train Cluster Loss": cluster_cost.item(),
                    "Train Separation Loss": separation_cost.item(),
                    "Train Cross Atten Loss": cross_coatten_cost.item(),
                    "Train Inte Align Loss": inte_align_cost.item(),
                    "Train Pred Align Loss": pred_align_cost.item(),
                    "Train Acc": n_correct / (n_examples + 0.000001) * 100,
                })

        del img_comb
        del target
        del logits
        del predicted
        del min_distances

    end = time.time()

    all_targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    all_auc = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions[:, i]) for i in range(model.module.num_classes - 1)],
    )
    mean_auc = all_auc.mean()


    print('##############TRAIN################')
    print('AUC:', np.around(all_auc, 4) * 100)
    print('Mean AUC:', mean_auc * 100)
    print('##############TRAIN################')


    log('\ttime: \t{0}'.format(end - start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tAUC: \t\t{0}%'.format(np.around(all_auc, 4) * 100))
    log('\tMean AUC: \t\t{0}%'.format(mean_auc * 100))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))

    wandb.log({
        "Train AUC": mean_auc * 100,
    })

    return mean_auc


def _testing(model, dataloader, optimizer=None, train_last=False, class_specific=True,  coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    predictions = []
    all_targets = []

    for i, (image, label, _) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, _ = model.module.forward_infer(input)

            # compute loss
            # cross_entropy = F.binary_cross_entropy(output, target)
            # cross_entropy = _binary_cross_entropy(output, target).mean()
            output_fg = output[:, 0:model.module.num_classes-1]
            output_bg = output[:, -1].unsqueeze(1).repeat(1, model.module.num_classes-1)  # no finding is the last prototypes
            output_new = torch.stack((output_bg, output_fg), dim=-1)
            cross_entropy = F.cross_entropy(output_new.permute(0, 2, 1), target[:, 0:model.module.num_classes-1])

            if class_specific:
                cluster_cost, separation_cost = cluster_sep_loss_fn(model, min_distances, label)
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                separation_cost = torch.mean(min_distance)

            predictions.append(F.softmax(output_new.data, dim=2)[:, :, 1].cpu().numpy())
            all_targets.append(label.numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

        del input
        del target
        del output
        del min_distances

    end = time.time()

    predictions = np.concatenate(predictions, axis=0)   # [N, 14] Probs of only the 14 foreground (disease) classes
    all_targets = np.concatenate(all_targets, axis=0)
    
    # evaluation statistics   
    predicted = 1.0 * (predictions > 0.5)
    n_examples = all_targets.shape[0] * all_targets[:, 0:model.module.num_classes].shape[1]
    print("👉 predicted shape:", predicted.shape)
    print("👉 all_targets shape:", all_targets.shape)
    print("👉 model.module.num_classes:", model.module.num_classes)
    print("👉 all_targets sliced shape:", all_targets[:, 0:model.module.num_classes].shape)

    n_correct += (predicted == all_targets[:, 0:model.module.num_classes]).sum().item()

    all_auc = np.asarray(
        [roc_auc_score(all_targets[:, i], predictions[:, i]) for i in range(model.module.num_classes - 1)],
    )
    mean_auc = all_auc.mean()

    print('##############TEST################')
    print('AUC:', np.around(all_auc, 4) * 100)
    print('Mean AUC:', mean_auc * 100)
    print('##############TEST################')

    max_f1s = []
    Accs = []
    for i in range(model.module.num_classes - 1):
        gt_np = all_targets[:, i]
        pred_np = predictions[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        Accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))

    f1_avg = np.array(max_f1s).mean()
    acc_avg = np.array(Accs).mean()
    print('$$$$$$$ Average F1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
    print('$$$$$$$ Average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))


    log('\ttime: \t{0}'.format(end - start))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tAUC: \t\t{0}%'.format(np.around(all_auc, 4) * 100))
    log('\tMean AUC: \t\t{0}%'.format(mean_auc * 100))
    log('\tMean F1: \t\t{0}%'.format(f1_avg * 100))
    log('\tMean ACC: \t\t{0}%'.format(acc_avg * 100))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))


    wandb.log({
            "Test AUC": mean_auc * 100,
        })

    # T_test = True
    T_test = False
    if T_test:
        sample_size = len(predictions)
        index_array = np.arange(sample_size)
        for i in range(1000):
            sampled_index = np.random.choice(index_array, sample_size, replace=True)
            if i == 0:
                print(sampled_index)
            all_auc = np.asarray(
                [roc_auc_score(all_targets[sampled_index, ccc], predictions[sampled_index, ccc]) for ccc in range(model.module.num_classes - 1)],
            )
            mean_auc = all_auc.mean()
            print(mean_auc)

            if i == 1000 - 1:
                print(sampled_index)

    return mean_auc


def train(model, dataloader, optimizer, warmup=False, train_last=False, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    if warmup:
        return _training_warmup(model=model, dataloader=dataloader, optimizer=optimizer, train_last=train_last,
                                class_specific=class_specific, coefs=coefs, log=log)
    else:
        return _training(model=model, dataloader=dataloader, optimizer=optimizer, train_last=train_last,
                         class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _testing(model=model, dataloader=dataloader, optimizer=None,  class_specific=class_specific, log=log)


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint')


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    log('\tlast layer')
