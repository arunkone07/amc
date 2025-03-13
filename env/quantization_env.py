import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from env.rewards import *
import math
import numpy as np
import copy


class QuantizationEnv:
    """
    Env for quantization search
    """
    def __init__(self, model, checkpoint, data, args, bitwidth= 8, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.quantizable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]

        # save options
        self.model = model
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.bitwidth = bitwidth
        
        # options from args
        self.args = args
        self.lbound = 1  # Lower bound for bits
        self.rbound = 16  # Upper bound for bits

        self.use_real_val = args.use_real_val

        self.n_calibration_batches = args.n_calibration_batches
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.curr_size = 0
        self.export_model = export_model
        self.use_new_input = use_new_input
        
        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self.n_quantizable_layer = len(self.quantizable_idx)

        # build embedding (static part)
        self._build_state_embedding()

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.org_model_size = self._calculate_model_size(self.model)
        print('=> original model size: {:.4f} bits'.format(self.org_model_size))
        # print(args.reward)
        self.reward = eval(args.reward) # Assumes reward function is defined in args.reward

        self.best_reward = -math.inf
        self.best_strategy = None

    def _init_data(self):
        # split the train set into train + val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)
        if self.use_real_val:
            print('*** USE REAL VALIDATION SET!')
            
    def _build_index(self):
        self.quantizable_idx = []
        self.quantizable_ops = []
        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.quantizable_ops.append(m)

        print('=> Quantizable layer idx: {}'.format(self.quantizable_idx))
        
    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            elif type(m) == nn.Linear:
                this_state.append(i)  # index
                this_state.append(1)  # layer type, 1 for fc
                this_state.append(m.in_features)  # in channels
                this_state.append(m.out_features)  # out channels
                this_state.append(0)  # stride
                this_state.append(1)  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size

            # this 2 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(self.rbound) # bits
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)
        self.layer_embedding = layer_embedding
        
    def _calculate_model_size(self, model):
        size = 0
        for m in model.modules():
            if type(m) in self.quantizable_layer_types:
                size += np.prod(m.weight.size()) * 32 # 32 bits for float.
        return size
    
    def _calculate_quantized_model_size(self):
        size = 0
        # strategy_index = 0
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                size += np.prod(m.weight.size()) * self.strategy[-1]
                # strategy_index += 1
        return size
    
    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return top1.avg
        elif self.acc_metric == 'acc5':
            return top5.avg
        else:
            raise NotImplementedError
    
    def set_export_path(self, path):
        self.export_path = path
        
    def _quantize_model(self, strategy):
        size = 0
        idx = 0
        for m in self.model.modules():
            if type(m) in self.quantizable_layer_types:
                bits = strategy[idx]
                idx += 1
                size += self._quantize_layer(m, bits)
        self.curr_size = size
    
    def _quantize_layer(self, layer, bitwidth):
        # m_list = list(self.model.modules())
        # layer = m_list[layer_index]
        size = np.prod(layer.weight.size()) * 32
        if type(layer) in self.quantizable_layer_types:
            if bitwidth != 32: #dont quantize if already 32 bits.
                layer.weight.data = self._quantize_tensor(layer.weight.data, bitwidth)
                size = size * bitwidth / 32
        return size

    def _quantize_tensor(self, tensor, bits):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = (max_val - min_val) / (2**bits - 1)
        if scale == 0:
            return torch.zeros_like(tensor)
        quantized_tensor = torch.round((tensor - min_val) / scale)
        dequantized_tensor = quantized_tensor * scale + min_val
        return dequantized_tensor
    
    def step(self, action):
        # Quantize and get the corresponding statistics.
        # print(action)
        action = self._action_wall(action) # Convert continuous action to discrete bits
        # self.curr_size += self._quantize_layer(self.quantizable_idx[self.cur_ind], action)
        self.strategy.append(action)  # save action to strategy

        
        # all the actions are made
        if self._is_final_layer():
            assert len(self.strategy) == len(self.quantizable_idx)
            self._quantize_model(self.strategy) # Quantize the model
            current_size = self.curr_size
            acc_t1 = time.time()
            acc = self._validate(self.val_loader, self.model)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            bit_sum = np.sum(self.strategy)
            compression_ratio = self.org_model_size / current_size
            info_set = {'compression_ratio': compression_ratio, 'accuracy': acc, 'strategy': self.strategy.copy()}
            reward = self.reward(self, acc, compression_ratio)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, bit_sum: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, bit_sum, compression_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            if self.export_model:  # export state dict
                torch.save(self.model.state_dict(), self.export_path)
                return None, None, None, None
            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-2] = self._cur_reduced(self.strategy) / self.org_model_size # reduced
        self.layer_embedding[self.cur_ind][-1] = self.strategy[-1] # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set
    
    def reset(self):
        # restore env by loading the checkpoint
        self.model.load_state_dict(self.checkpoint)
        self.cur_ind = 0
        self.curr_size = 0
        self.strategy = []  # quantization strategy
        # reset layer embeddings
        self.layer_embedding[:, -1] = self.rbound # set to max bits at beginning
        self.layer_embedding[:, -2] = 0.
        obs = self.layer_embedding[0].copy()
        obs[-2] = 0.
        self.val_time = 0
        return obs
    
    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1
    
    def _action_wall(self, action):
        # if action < 0.25:
        #     return 1
        if action < 0.25:
            return 2
        elif action < 0.5:
            return 4
        else:
            return 8
    
    def _cur_reduced(self, strategy):
        # return the reduced weight
        # print(self.strategy)
        quantized_size = self._calculate_quantized_model_size()
        reduced = self.org_model_size - quantized_size
        return reduced