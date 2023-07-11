import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def Entropy(input_):
    return torch.sum(-input_ * torch.log(input_ + 1e-5), dim=1)

def grl_hook(coeff):
    def fun1(grad):
        return - coeff * grad.clone()
    return fun1

def DANN(features, ad_net):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if dc_target.is_cuda:
        dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    feature = input_list[0]
    softmax_output = input_list[1].detach()
    
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if feature.is_cuda:
        dc_target = dc_target.cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy * source_mask

        target_mask = torch.ones_like(entropy)
        target_mask[:feature.size(0)//2] = 0
        target_weight = entropy * target_mask

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def HAFN(features, weight_L2norm, radius):
    
    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    return weight_L2norm * (features.norm(p=2, dim=1).mean() - radius) ** 2

def SAFN(features, weight_L2norm, deltaRadius):

    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    radius = features.norm(p=2, dim=1).detach()

    assert radius.requires_grad == False, 'radius\'s requires_grad should be False'
    
    return weight_L2norm * ((features.norm(p=2, dim=1) - (radius+deltaRadius)) ** 2).mean()

def Reweighted_MMD(feature, source_info, target_info):

    '''
    Paper Link : https://arxiv.org/pdf/1904.11150.pdf
    '''

    def k(x, y, sigma=1):
        distance = torch.squeeze(torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)(x.unsqueeze(0), y.unsqueeze(0)))
        # distance = distance * distance
        return torch.exp(-distance/(2*sigma*sigma))

    assert feature.size(0)%14==0, "Feature Size is wrong."
    assert len(source_info)==8 and source_info[0]==(source_info[1]+source_info[2]+source_info[3]+source_info[4]+source_info[5]+source_info[6]+source_info[7]), "Source Info is wrong."
    assert len(target_info)==8 and target_info[0]==(target_info[1]+target_info[2]+target_info[3]+target_info[4]+target_info[5]+target_info[6]+target_info[7]), "Target Info is wrong."

    N_s, N_t, numOfPerClass = feature.size(0)//2, feature.size(0)//2, feature.size(0)//14

    # Compute Alpha
    alpha = [ (target_info[i+1]/target_info[0])/(source_info[i+1]/source_info[0]) for i in range(7)]

    # Compute Alpha MMD Loss
    alpha_MMD = 0
   
    for i in range(N_s-1):
        for j in range(i+1, N_s):
            alpha_MMD+=k(feature[i],feature[j]) * alpha[i//numOfPerClass] * alpha[j//numOfPerClass] / (N_s * (N_s-1))

    for i in range(N_t-1):
        for j in range(i+1, N_t):
            alpha_MMD+=k(feature[i+N_s],feature[j+N_s]) / (N_t * (N_t-1))

    for i in range(N_s):
        for j in range(N_t):
            alpha_MMD-=k(feature[i], feature[j+N_s]) * alpha[i//numOfPerClass] * 2 / (N_s * N_t)

    # Compute Conditional MMD Loss
    conditional_MMD = 0
    
    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range(i+1, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i], feature[j])/ (numOfPerClass * (numOfPerClass-1))

    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range(i+1, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i+N_s], feature[j+N_s])/ (numOfPerClass * (numOfPerClass-1))

    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1)* numOfPerClass):
            for j in range(Class * numOfPerClass, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i], feature[j+N_s]) * 2 / (numOfPerClass * numOfPerClass)

    return alpha_MMD, conditional_MMD


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            #with torch.no_grad():
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            #torch.cuda.empty_cache()
            return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)



def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss