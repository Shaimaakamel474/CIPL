import os
import torch

def save_model_w_condition(model, model_dir, model_name, auc, target_auc, log=print):
    '''
    model: this is not the multigpu model
    '''
    if auc > target_auc:
        log('\tabove {0:.2f}%'.format(target_auc * 100))
        torch.save(model.state_dict(), os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(auc)))
        # torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(auc)))