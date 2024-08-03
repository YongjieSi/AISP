import logging
import os
import random
import numpy as np
import torch
from torch import nn, tensor
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MultiBranchCosineIncrementalNet,AEFEVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal


# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8
criterion  = nn.CrossEntropyLoss()


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # if 'adapter' not in args["backbone_type"]:
        #     raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = AEFEVitNet(args, True)
        self. init_lr=args["init_lrate"]
        self.weight_decay=args["init_weight_decay"] if args["init_weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self.task_sizes = []


    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()



    def incremental_train(self, data_manager,layers=None):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['train_batch_size'], shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['test_batch_size'], shuffle=False, num_workers=num_workers)
        self.args["mode"] = "train"
        self._train(self.train_loader, self.test_loader, layers)

    def _train(self, train_loader, test_loader, layers):

        self._network.to(self._device)

        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            self.construct_dual_branch_network()
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD([{'params': self._network.backbones[1:].parameters(), 'lr': self.args["init_lrate"]},
                                    {'params': self._network.fc1.parameters(), 'lr': self.args["fc_lrate"]},
                                    {'params': self._network.fc.parameters(), 'lr': self.args["fc_lrate"]},
                                    ], momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['init_epoch'], eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler,layers,use_margin=self.args['use_margin'])

        else:
            optimizer = optim.SGD([
                {'params': self._network.backbones[1].parameters(), 'lr': self.args["init_lrate"]},
                {'params': self._network.backbones[3].parameters(), 'lr': self.args["init_lrate"]},
                {'params': self._network.backbones[4].parameters(), 'lr': self.args["init_lrate"]},
                {'params': self._network.fc1.parameters(), 'lr': self.args["fc_lrate"]},
                {'params': self._network.fc.parameters(), 'lr': self.args["fc_lrate"]},
                                    ], momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['new_epochs'], eta_min=self.min_lr)
            self._update_representation(train_loader, test_loader, optimizer, scheduler,use_margin=self.args['use_margin'],use_kd=self.args['use_kd'])
        self._compute_class_mean(self.data_manager, check_diff=False, oracle=False)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[1]
            assert ori_classes==self._known_classes
            new_class_means = np.zeros(((12-self.args['blocks'])*2,self._total_classes, self.feature_dim))
            new_class_means[:,:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros(((12-self.args['blocks'])*2,self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:,:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros(((12-self.args['blocks'])*2,self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros(((12-self.args['blocks'])*2,self._total_classes, self.feature_dim, self.feature_dim))
        for class_idx in range(self._known_classes, self._total_classes):
            if self.args['model_name'] == 'fcac':
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='base', ret_data=True)
            else:
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='train', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=25, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = torch.from_numpy(vectors)
            # vectors = vectors.squeeze(0)
            vectors = vectors.reshape((12-self.args['blocks'])*2,-1,768)
            gerneral_vectors = vectors[:12-self.args['blocks'],:,:]
            specific_vectors = vectors[12-self.args['blocks']:,:,:]
            for index, layer_data in enumerate(gerneral_vectors):
                gerneral_class_mean = torch.mean(layer_data, axis=0)
                gerneral_class_cov = torch.cov((layer_data).T)+torch.eye(gerneral_class_mean.shape[-1])*1e-4
                self._class_means[index, class_idx, :] = gerneral_class_mean
                self._class_covs[index, class_idx, ...] = gerneral_class_cov
            for index, layer_data in enumerate(specific_vectors):
                specific_class_mean = torch.mean(layer_data, axis=0)
                specific_class_cov = torch.cov((layer_data).T)+torch.eye(specific_class_mean.shape[-1])*1e-4
                self._class_means[index+12-self.args['blocks'], class_idx, :] = specific_class_mean
                self._class_covs[index+12-self.args['blocks'],  class_idx, ...] = specific_class_cov


    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler,layers, use_margin=False):

        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs,layers)["logits"]
                if use_margin:
                    angular_criterion = CosFaceLoss(s=self.args["margin_temp"], m=self.args["positive_m"])
                    loss = angular_criterion(logits, targets) # new
                else:
                    loss= F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['init_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler,use_margin=False,use_kd=False, use_old_data = True):

        prog_bar = tqdm(range(self.args["new_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            if use_old_data:
                old_data, old_label = self.get_old_data()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, new_targets = inputs.to(self._device), targets.to(self._device)
                if use_old_data:
                    logits = self._network(inputs,old_data)["logits"]
                    targets = torch.cat((new_targets,old_label[0]))
                else:
                    logits = self._network(inputs)["logits"]
                    targets = new_targets
                if use_margin:
                    angular_criterion = CosFaceLoss(s=self.args["margin_temp"], m=self.args["positive_m"])
                    loss_clf = angular_criterion(logits, targets) # new
                else:
                    loss_clf = F.cross_entropy(logits, targets)
                if use_kd:
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs,old_data)["logits"],
                        self.args["T"],
                    )
                    loss = loss_clf + loss_kd
                else:
                    loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                # test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["new_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )

            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["new_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def get_old_data(self,layer_embedding=True):
        sampled_data = []
        sampled_label =[]
        num_sampled_pcls = 5
        for layer in range(0,(12-self.args['blocks'])*2):
            for c_id in range(self._known_classes):
                t_id = c_id//5
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = torch.tensor(self._class_means[layer][c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self._class_covs[layer][c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id]*num_sampled_pcls)
        old_data = torch.stack(sampled_data).view((12-self.args['blocks'])*2, self._known_classes * num_sampled_pcls, -1)
        old_label = torch.tensor(sampled_label).view((12-self.args['blocks'])*2, self._known_classes * num_sampled_pcls)
        old_label =old_label.to(self._device)
        return old_data, old_label.to(self._device)


class CosFaceLoss(nn.Module):
    def __init__(self, s=32.0, m=0.5):
        super(CosFaceLoss, self).__init__()
        self.s = s  # scale factor
        self.m = m  # margin

    def forward(self, input, labels):
        one_hot = torch.zeros(input.size(), device=input.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        input = input - one_hot * self.m
        input = input * self.s
        loss = F.cross_entropy(input, labels)
        return loss

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
