from __future__ import print_function

import argparse
import os
import random
import sys
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import traceback
from collections import OrderedDict
from lion_pytorch import Lion


# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

# dataset
# from feeder.feeder import LandscapeDataset


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Landscape Classification Network')
    
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/test.yaml',
        help='path to the configuration file')

    # processor

    parser.add_argument(
        '--save-score',
        type=bool,
        default=False,
        help='if true, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=10,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=bool,
        default=True,
        help='print logging or not')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=2,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=20,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')

    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--adjust_lr', type=bool, default=True)


    # force rerun

    return parser

class Processor():

    def __init__(self, arg):
        self.arg = arg


        # Tensorboard
        arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
        self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')


        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.global_step = 0
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        
        


    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        self.model = Model(**self.arg.model_args)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            weights = torch.load(self.arg.weights)
            self.model.load_state_dict(weights)


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
            
        elif self.arg.optimizer == 'Lion':
            self.optimizer = Lion(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr



    def train(self, epoch, save_model=False):
        self.model.train()
        loader = self.data_loader['train']
        if self.arg.adjust_lr:
            self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

        process = tqdm(loader, ncols=40)

        for idx, (path, images, captions, label) in enumerate(process):
            images = images.cuda()
            captions = captions.cuda()
            label = label.cuda()



            # forward
            output = self.model(images, captions)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())


            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # break

        print('\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))



        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        print('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            acc_value = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)


            for idx, (path, images, captions, label) in enumerate(process):
                label_list.append(label)

                with torch.no_grad():
                    images = images.cuda()
                    captions = captions.cuda()
                    label = label.cuda()
                    output = self.model(images, captions)

                    loss = self.loss(output , label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                    acc = torch.mean((predict_label == label.data).float())

                    print(predict_label, label)
                    print(acc_value)
                    acc_value.append(acc.data.item())
                

                # break

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = np.mean(acc_value) * 100
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('\tAccuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            print('\n')

            self.val_writer.add_scalar('loss', loss, self.global_step)
            self.val_writer.add_scalar('acc', accuracy, self.global_step)


            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        print('Parameters:\n{}\n'.format(str(vars(self.arg))))
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'# Parameters: {count_parameters(self.model)}')
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

            self.train(epoch, save_model=save_model)

            self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

        # test the best model
        weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)

        self.eval(epoch=0, save_score=True, loader_name=['test'])


        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Best accuracy: {self.best_acc}')
        print(f'Epoch number: {self.best_acc_epoch}')
        print(f'Model name: {self.arg.work_dir}')
        print(f'Model total number of params: {num_params}')
        print(f'Weight decay: {self.arg.weight_decay}')
        print(f'Base LR: {self.arg.base_lr}')
        print(f'Batch Size: {self.arg.batch_size}')
        print(f'Test Batch Size: {self.arg.test_batch_size}')
        print(f'seed: {self.arg.seed}')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    arg.work_dir = arg.work_dir
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()