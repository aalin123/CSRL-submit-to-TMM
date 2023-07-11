import torch
import torch.nn as nn
import torch.nn.functional as F
from attention1 import *  
import torch.utils.model_zoo as model_zoo

from CrossdomainTrans import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ['downsample']
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class Attentiongabol(nn.Module):
    __constants__ = ['downsample']
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Attentiongabol, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.conv2 =SKAttention(planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)
        #self.mrcbam = MR_CBAM(planes,16)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        #out = self.mrcbam(out)
        out = self.cbam(out)
        
       
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block_b,  block_a, layers, num_classes=7):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        self.layer1 = self._make_layer(block_b, 64, 64, layers[0])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_b, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_b, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #self.fc_1 = nn.Linear(512*block_b.expansion, num_classes)

        # bottleneck_list = [nn.Linear(512*block_b.expansion, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5)]
        # self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        # self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        # self.bottleneck_layer[0].bias.data.fill_(0.1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        #x1 = self.bottleneck_layer(x)


        #pred = self.fc_1(x)
       

        return x#pred#, x 



class TransNet(nn.Module):

    def __init__(self, args):
        super(TransNet, self).__init__()
        self.sharedNet = ShareResNet(args)
        self.cls_fc = nn.Linear(512, args.class_num)
        # self.encoder_layers = STransformerEncoderLayer(512,2,512,0.5)
        # self.STencoder_layers = CrossTransformerEncoderLayer(512,2,512,0.5)
        
    def forward(self, source, target,Trag):
       
        source1 = self.sharedNet(source)
        source2 = source1
        if self.training == True:
            target1 = self.sharedNet(target)
            target = target1
            # SSource = source1.unsqueeze(1)
            # STarget = target1.unsqueeze(1)  
            # SSource1 = self.encoder_layers(SSource).squeeze(1)
            # TSource = self.STencoder_layers(SSource,STarget).squeeze(1)
            # Transfea=torch.cat((SSource1,TSource),0)
            # source1=torch.cat((source1,Transfea),0)
            if Trag== True:
                source1=torch.cat((source1,target),0)
            
        
        pred = self.cls_fc(source1)

        return pred, source2, target 

class TransNet_Dual(nn.Module):

    def __init__(self, args):
        super(TransNet_Dual, self).__init__()
        self.sharedNet = ShareResNet(args)
        self.cls_fc = nn.Linear(512, args.class_num)
        self.encoder_layers = STransformerEncoderLayer(512,2,512,0.5)
        self.STencoder_layers = CrossTransformerEncoderLayer(512,2,512,0.5)
        # self.Tencoder_layers = STransformerEncoderLayer(512, 4, 512, 0.5)
        # self.TSencoder_layers = CrossTransformerEncoderLayer(512, 2, 512, 0.5)
        # bottleneck_list1 = [nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5)]#,
        # self.bottleneck_layer = nn.Sequential(*bottleneck_list1)
        # self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        # self.bottleneck_layer[0].bias.data.fill_(0.1)
    
    
    def forward(self, source, target, Trag):
        source1 = self.sharedNet(source) #32x512
        source = source1 
        if self.training == True:
            target1 = self.sharedNet(target)
            target = target1
            SSource = source1.unsqueeze(1) 
            TTarget = target1.unsqueeze(1)  
            SSource1 = self.encoder_layers(SSource).squeeze(1)
            STource = self.STencoder_layers(SSource,TTarget).squeeze(1)
            Transfea=torch.cat((SSource1,STource),0)
            source1=torch.cat((source1,Transfea),0)
            if Trag==True:
                source1=torch.cat((source1,target1),0)
                TTarget1 =self.encoder_layers(TTarget).squeeze(1)
                TSarget = self.STencoder_layers(TTarget,SSource).squeeze(1)
                transfeaTS=torch.cat((TTarget1,TSarget),0)
                source1=torch.cat((source1,transfeaTS),0)
                
        pred = self.cls_fc(source1)

        return pred, source, target

def ShareResNet(args):

    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',}

    if  args.Backbone == 'ResNet18':
        model = ResNet(block_b=BasicBlock, block_a=Attentiongabol, layers=[2, 2, 2, 2])
    elif args.Backbone == 'ResNet50':
        model = ResNet(block_b=Bottleneck, block_a=Bottleneck, layers=[3, 4, 6, 3])
    if args.Resume_Model != 'None':
        if args.Resume_Model =='imagenet':
            print('Resume Model: {}'.format(args.Resume_Model))
            if  args.Backbone == 'ResNet18':
                model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
            elif args.Backbone == 'ResNet50':
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
        else:
            print('Resume Model: {}'.format(args.Resume_Model))
            checkpoint = torch.load(args.Resume_Model, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # Save GPU Memory
            del checkpoint
            torch.cuda.empty_cache()
    else:
        print('No Resume Model')
   
    return model

def ResNet18():
    return ResNet(block_b=BasicBlock, block_a=Attentiongabol, layers=[2, 2, 2, 2])


def ResNet50():
    return ResNet(block_b=Bottleneck, block_a=Bottleneck, layers=[3, 4, 6, 3])