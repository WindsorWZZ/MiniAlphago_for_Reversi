import hashlib
import json
import os
import sys
from logging import getLogger

# nets
import torch
import torch.nn as nn
import torch.nn.functional as F

# config
from args import Config
import numpy as np


# logger
logger = getLogger(__name__)

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def _param_init(m):
    if isinstance(m, nn.Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)

class OthelloModel(nn.Module):
    def __init__(self, config:Config):
        # init
        super(OthelloModel, self).__init__()
        self.best = None
        self.config = config
        self.device = torch.device('cpu') if config.opts.do == 'self_play'  else self.config.opts.device
        self.first_conv = self.build_residual_block1(2, self.config.model.cnn_filter_size)
        self.conv_blocks = []
        for i in range(self.config.model.res_layer_num):
            temp = self.build_residual_block2(self.config.model.cnn_filter_size).to(self.device)
            self.add_module("f{i}",temp)
            self.conv_blocks.append(temp)
        self.policy_conv = self.build_residual_block1(self.config.model.cnn_filter_num ,1)
        self.value_conv = self.build_residual_block1(self.config.model.cnn_filter_num ,1)
        self.policy_linear = nn.Linear(25600, 64)
        self.value_linear_1 = nn.Linear(25600, self.config.model.value_fc_size)
        self.value_linear_2 = nn.Linear(256, 1)

        # init
        for p in self.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    _param_init(pp)
            else:
                _param_init(p)

        # cuda
        self.to(self.device)

    def forward(self, input:torch.FloatTensor):
        # change input
        if type(input)!=torch.FloatTensor:
            if type(input)==np.ndarray:
                input = torch.FloatTensor(input)
        input = input.to(self.device)
        # get config
        model_config = self.config.model

        # process
        x = self.first_conv(input)
        for i in range(model_config.res_layer_num):
            forward = self.conv_blocks[i](x)
            x = x + forward
            x = F.relu(x)
        res_out = x

        # for policy output
        x = self.policy_conv(res_out)
        x = x.view(x.shape[0], -1)
        policy_out = self.policy_linear(x)
        policy_out = F.log_softmax(policy_out, dim=1)

        # for value output
        x = self.value_conv(res_out)
        x = x.view(x.shape[0], -1)
        x = self.value_linear_1(x)
        x = F.relu(x)
        value_out = self.value_linear_2(x)
        value_out = torch.tanh(value_out)
        if self.config.opts.do=='self_play' or self.config.opts.do=='play_gui' or self.config.opts.do=='make_best':
            return policy_out.cpu().data.numpy(), value_out.cpu().data.numpy()
        elif self.config.opts.do == 'optimize':
            return policy_out.cpu(), value_out.cpu()


    def loss(self, p_pred, v_pred, p_true, v_true):
        if type(p_pred)!=torch.FloatTensor:
            p_pred = torch.FloatTensor(p_pred)
        if type(p_true)!=torch.FloatTensor:
            p_true = torch.FloatTensor(p_true)
        if type(v_pred)!=torch.FloatTensor:
            v_pred = torch.FloatTensor(v_pred)
        if type(v_true)!=torch.FloatTensor:
            v_true = torch.FloatTensor(v_true)
        return torch.sum(torch.sum(-p_true*p_pred, -1),0)+F.mse_loss(v_pred, v_true)

    def build_residual_block1(self, input_dim, kernel_size):
        """1 res blocks"""
        model_config = self.config.model
        res = nn.Sequential(nn.Conv2d(in_channels=input_dim,
                                       out_channels=model_config.cnn_filter_num,
                                       kernel_size=kernel_size,
                                       padding=1
                                       ),
                             nn.BatchNorm2d(model_config.cnn_filter_num),
                             nn.ReLU())
        return res

    def build_residual_block2(self, kernel_size):
        """2 res blocks"""
        model_config = self.config.model
        res = nn.Sequential(nn.Conv2d(in_channels=model_config.cnn_filter_num,
                      out_channels=model_config.cnn_filter_num,
                      kernel_size=kernel_size,
                      padding=1
                      ),
        nn.BatchNorm2d(model_config.cnn_filter_num),
        nn.ReLU(),
        nn.Conv2d(in_channels=model_config.cnn_filter_num,
                      out_channels=model_config.cnn_filter_num,
                      kernel_size=kernel_size,
                      padding=1
                      ),
        nn.BatchNorm2d(model_config.cnn_filter_num))
        return res

    def load(self, weight_path):
        # load
        if os.path.exists(weight_path):
            logger.debug(f"loading model from {weight_path}")
            self.load_state_dict(torch.load(weight_path))
            return True
        else:
            logger.debug(f"model files does not exist at {weight_path}")
            return False

    def save(self, weight_path):
        # save
        logger.debug(f"save model to {weight_path}")
        torch.save(self.state_dict(), weight_path)
        return True

    # def fetch_digest(self, weight_path):

if __name__ == "__main__":
    from main import config
    model = OthelloModel(config=config)

    #test 1
    # x = torch.randn(1,2,8,8)
    # print(model.loss(*model(x),torch.tensor([1]).long(), torch.tensor([[5]]).float()))

    #test 2
    # print(model.state_dict())

    #test 3
    import h5py
    f = h5py.File("../data/model/best_weight/model_best_weight.h5", 'r')  # 打开h5文
    for g in f.attrs:
        print(g)

    # for key in f.keys():
    #     print(f[key].name)
    #     try:
    #         print(type(f[key]))
    #         print(f[key].value)
    #     except:
    #         pass

