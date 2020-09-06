from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import TRPN
import shutil
import os
import random
from WRN_train import ModelTrainer
from collections import OrderedDict

if __name__ == '__main__':

    tt.arg.test_model = 'WRN-Pretrained-tiered-5-1' if tt.arg.test_model is None else tt.arg.test_model
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = "/home/jovyan/16061167/open_code2/WRN_tired_DATA/" 
    tt.arg.dataset = 'tiered' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.meta_batch_size = 40 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    #tt.arg.transductive = False if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus
    tt.arg.features = True

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    tt.arg.emb_size = 640
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 10
    tt.arg.log_step = 100

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    gcn_module = TRPN(n_feat=tt.arg.emb_size, n_queries=tt.arg.num_ways_test * 1)

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'tiered':
        test_loader = TieredImagenetLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')


    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(
                           gcn_module=gcn_module,
                           data_loader=data_loader)


    checkpoint = torch.load(tt.arg.test_model + '/model_best.pth.tar')
    pretrained_dict = checkpoint['gcn_module_state_dict']
    new_state_dict = OrderedDict()
    ## load parameters to a single gpu
    for k, v in pretrained_dict.items():
        name = k[7:] # remove 'module'.
        new_state_dict[name] = v
    model_dict = tester.gcn_module.state_dict()
    tester.gcn_module.load_state_dict(new_state_dict)
    print("load pre-trained enc_nn done!")
    print("load pre-trained egnn done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']
    print(tester.val_acc)
    print(tester.global_step)


    tester.eval(partition='test')





