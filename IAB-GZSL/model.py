import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#from utils import set_gpu
import numpy as np
import os
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NAA(nn.Module):
    def __init__(self,opt, group, data):
        super(NAA, self).__init__()
        self.opt = opt
        self.group = group
        beta_shape = [1,len(group)]
        if opt.train_beta:
            self.betas = nn.Parameter(2e-3 * torch.rand(beta_shape), requires_grad=True)
        else:
            self.betas = nn.Parameter(0.0 * torch.rand(beta_shape), requires_grad=False)
        self.Lp1 = len(self.group)+1
        self.data = data

    def beta_finetune(self, bool):
        self.betas.requires_grad = bool

    def prepare_attri_label(self, attribute, classes):
        classes_dim = classes.size(0)
        attri_dim = attribute.shape[1]
        output_attribute = torch.FloatTensor(classes_dim * self.Lp1, attri_dim)
        for i in range(classes_dim):
            output_attribute[i * self.Lp1:i * self.Lp1 + self.Lp1] = attribute[classes[i] * self.Lp1 : classes[i] * self.Lp1 + self.Lp1]
        return torch.transpose(output_attribute, 1, 0)

    def forward(self,attribute):
        multy_attribute = torch.zeros(attribute.shape[0] * self.Lp1, self.opt.att_size).to(self.opt.device)
        for i in range(len(self.data.allclasses)):
            multy_attribute[i * self.Lp1:i * self.Lp1 + self.Lp1] = attribute[i]
            for k, name in enumerate(self.group):
                if k != 0:
                    multy_attribute[k + (i * (len(self.group) + 1)) + 1, self.group[name]] = self.betas[0,k-1]
        multy_attribute = F.normalize(multy_attribute, dim=1)
        attribute_zsl = self.prepare_attri_label(multy_attribute, self.data.unseenclasses).to(self.opt.device)
        attribute_seen = self.prepare_attri_label(multy_attribute, self.data.seenclasses).to(self.opt.device)
        attribute_gzsl = torch.transpose(multy_attribute, 1, 0).to(self.opt.device)
        return attribute_zsl, attribute_seen, attribute_gzsl

class VARS(nn.Module):
    def __init__(self, cin,opt):
        super(VARS, self).__init__()
        self.activation_head = nn.Conv2d(cin, opt.att_size, kernel_size=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(opt.att_size)
        self.opt = opt
    def forward(self,x):
        N, C, H, W = x.size()
        a = self.activation_head(x)
        cam = torch.sigmoid(self.bn_head(a))
        ccam_ = cam.reshape(N, self.opt.att_size, H*W)
        x = x.reshape(N, C, H*W).permute(0, 2, 1).contiguous()
        fg_feats = torch.matmul(ccam_, x)/(H*W*self.opt.att_size)
        bg_feats = torch.matmul(1-ccam_, x)/(H*W*self.opt.att_size)
        fg_feats = fg_feats.sum(1)
        bg_feats = bg_feats.sum(1)
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), a

class Soft_Sort(nn.Module):
    def __init__(self,tau = 1.0,pow = 1.0):
        super().__init__()
        self.tau = tau
        self.pow = pow
    def forward(self,scores):
        scores = scores.unsqueeze(-1)
        sorted = torch.sort(scores, descending=True, dim=1)[0]
        #print(sorted)
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)
        return P_hat

class IAB(nn.Module):
    def __init__(self, opt, group_dic, data):
        super(IAB, self).__init__()
        resnet = models.resnet101()
        num_ftrs = resnet.fc.in_features
        num_fc_dic = {'cub':150, 'awa2': 1000, 'sun': 645,'flo':1000}

        if opt.dataset =='AWA2':
            num_fc = num_fc_dic['awa2']
        elif opt.dataset =='CUB':
            num_fc = num_fc_dic['cub']
        elif opt.dataset =='SUN':
            num_fc = num_fc_dic['sun']
        else:
            num_fc = 1000
        resnet.fc = nn.Linear(num_ftrs, num_fc)

        _, ext = os.path.splitext(opt.resnet_path)
        if opt.resnet_path == './pretrained_models/resnet101-5d3b4d8f.pth':
            state_dict = torch.load(opt.resnet_path)
            resnet.load_state_dict(state_dict)
            print('load resnet from ', opt.resnet_path)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        if ext == ".pth" and opt.resnet_path != None and opt.resnet_path != './pretrained_models/resnet101-5d3b4d8f.pth':
            state_dict = torch.load(opt.resnet_path)
            self.resnet.load_state_dict(state_dict)
            print('load resnet from ', opt.resnet_path)

        if opt.image_size == 224:
            self.kernel_size = 7
        else:
            self.kernel_size = 14
        self.softsort = Soft_Sort(tau=1,pow=1)
        self.prototype = CNZSLModel(opt.att_size, opt.latent_dim, opt.cls_dim)
        self.vars = VARS(opt.feat_dim,opt=opt)
        self.opt = opt

        self.NAA = NAA(opt, group_dic, data)
    def get_logits(self,visual_feature,visual_protos):
        logits = dict()
        visual_x_ns = self.opt.t * visual_feature / visual_feature.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        visual_protos_ns = self.opt.t * visual_protos / visual_protos.norm(dim=1, keepdim=True)

        visual_logits = visual_x_ns @ visual_protos_ns.t()
        diffs = (visual_feature.unsqueeze(1) - visual_protos.unsqueeze(0))
        squared_diffs = diffs ** 2
        l2_norms = squared_diffs.sum(2)
        l2_scores = -l2_norms

        logits['cos'] = visual_logits
        logits['l2'] = l2_scores
        return logits

    def get_logits_test(self,visual_feature,visual_protos):
        logits = dict()
        visual_x_ns = visual_feature / visual_feature.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        visual_protos_ns = visual_protos / visual_protos.norm(dim=1, keepdim=True)
        visual_logits = visual_x_ns @ visual_protos_ns.t()
        logits['cos'] = visual_logits
        return logits

    def train_forward(self, x, attribute):
        x = self.resnet[0:5](x)  # layer 1
        x = self.resnet[5](x)  # layer 2
        x = self.resnet[6](x)  # layer 3
        x = self.resnet[7](x)  # layer 4

        top_k = dict()
        _, attribute, _ = self.NAA(attribute)
        prototypes = self.prototype(attribute.T)

        fg_feature, bg_feature, map = self.vars(x)
        logits = self.get_logits(fg_feature, prototypes)
        top_k['cos_'] = logits['cos']
        top_k['cos'] = self.softsort(logits['cos'])
        top_k['l2'] = self.softsort(logits['l2'])
        top_k['cos'] = top_k['cos'][:, :self.opt.gamma, :].sum(1)
        top_k['l2'] = top_k['l2'][:, :self.opt.gamma, :].sum(1)

        psi_x = F.max_pool2d(map, kernel_size=self.kernel_size)
        sim_map = self.get_logits(psi_x.view(-1,self.opt.att_size), attribute.T)
        sim_map['cos_'] = sim_map['cos']
        sim_map['l2'] = self.softsort(sim_map['l2'])
        sim_map['cos'] = self.softsort(sim_map['cos'])
        sim_map['l2'] = sim_map['l2'][:, :self.opt.gamma, :].sum(1)
        sim_map['cos'] = sim_map['cos'][:, :self.opt.gamma, :].sum(1)

        return top_k, sim_map, fg_feature, bg_feature, map

    def test_forward(self, x, attribute):
        x = self.resnet[0:5](x)  # layer 1
        x = self.resnet[5](x)  # layer 2
        x = self.resnet[6](x)  # layer 3
        x = self.resnet[7](x)  # layer 4

        if self.opt.NAA_test:
            _, _, attribute = self.NAA(attribute)
        prototypes = self.prototype(attribute.T)
        if self.opt.vars_test:
            feature, _, _ = self.vars(x)
        else:
            feature = F.avg_pool2d(x, kernel_size=self.kernel_size).view(-1, 2048)
        logits = self.get_logits_test(feature, prototypes)
        return logits['cos']

    def forward(self, x, attribute):
        if self.training:
            top_k, sim_map, fg_feature, bg_feature, map = self.train_forward(x, attribute)
            return top_k, sim_map, fg_feature, bg_feature, map
        else:
            with torch.no_grad():
                pred = self.test_forward(x,attribute)
            return pred

    def fine_tune(self, fine_tune_low = False, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune_low
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def vars_finetune(self, fine_tune=True):
        for p in self.vars.parameters():
            p.requires_grad = fine_tune


class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result


class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()

        self.USE_PROPER_INIT = True  # i.e. equation (10) from the paper
        self.USE_CLASS_STANDARTIZATION = True

        self.model = nn.Sequential(
            nn.Linear(attr_dim, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            # nn.ReLU(inplace=True)
        )
        if self.USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-1].weight.data.uniform_(-b, b)
    def forward(self, attrs):
        visual_protos = self.model(attrs)
        return visual_protos