import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import torch.nn.functional as F

class ISICNet(nn.Module):
    def __init__(self, features):
        super(ISICNet, self).__init__()
        self.features = features
        self.classifier = make_classifier("C1")
        self.output = make_output_layer()
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)        
        return x

    def _initialize_weights(self, stddev=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, stddev)
                m.bias.data.zero_()

    def get_model_parameters_dict(self):
        i = 0
        params = {}
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                name = "l." + str(i) + ".conv" 
                val = m.weight.data.cpu().numpy()
                # pytorch dimensions (out_channels, in_channels, kernel_size[0], kernel_size[1])
                # tensorflow dimensions [filter_height, filter_width, in_channels, out_channels]
                params[name + '.weight'] = np.transpose(val, (2,3,1,0))
                params[name + '.bias'] = m.bias.data.cpu().numpy()
            elif isinstance(m, nn.BatchNorm2d):
                name = "l." + str(i) + ".batchnorm"
                params[name + '.weight'] = m.weight.data.cpu().numpy()
                params[name + '.bias'] = m.bias.data.cpu().numpy()
                params[name + '.running_mean'] = m.running_mean.cpu().numpy()
                params[name + '.running_var'] = m.running_var.cpu().numpy()
            elif isinstance(m, nn.Linear):
                name = "l." + str(i) + ".linear"
                params[name + '.weight'] = np.transpose(m.weight.data.cpu().numpy())
                params[name + '.bias'] = m.bias.data.cpu().numpy()
            elif isinstance(m, nn.MaxPool2d):
                name = "l." + str(i) + ".maxpool"
                params[name] = 0
            elif isinstance(m, nn.ReLU):
                name = "l." + str(i) + ".relu"
                params[name] = 0
            i += 1
        return params

def make_classifier(classifier_str):
    if classifier_str == "C1":
        return nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),            
            nn.AvgPool2d(kernel_size=4),
        ) 
        

def make_output_layer():
    return nn.Sequential(nn.Linear(64, 3), 
                        )
                        
def make_layers(cfg, batch_norm=False, conv_size=3, pad_size=1):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_size, padding=pad_size)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers_rev2():
    cfg7 = 16
    cfg = ['M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M']
    layers = []
    in_channels = 3
    conv2d = nn.Conv2d(in_channels, cfg7, kernel_size=7, padding=3, stride=2)
    layers += [conv2d, nn.BatchNorm2d(cfg7), nn.ReLU(inplace=True)]
    in_channels = cfg7
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M']   
}


def ret1_bn(**kwargs):
    return ISICNet(make_layers(cfg['A'], batch_norm=True, conv_size=3, pad_size=1), **kwargs)

def ret2_bn(**kwargs):
    return ISICNet(make_layers_rev2(), **kwargs)

# Create a model feature extractor removing the last output layer
class model_explainable_ret1_bn(nn.Module):    
    def __init__(self, original_model):
        super(model_explainable_ret1_bn, self).__init__()
        self.rf = [1.,3.,5.,9.,13.,21.,29.,45.,61.,93.,125.,189.,253.,381.,509.,637.,640.,640.]
        self.rf3 = nn.Sequential(*list(original_model.features.children())[0:3])
        self.rf5 = nn.Sequential(*list(original_model.features.children())[3:6])
        self.rf9 = nn.Sequential(*list(original_model.features.children())[7:10])
        self.rf13 = nn.Sequential(*list(original_model.features.children())[10:13])
        self.rf21 = nn.Sequential(*list(original_model.features.children())[14:17])
        self.rf29 = nn.Sequential(*list(original_model.features.children())[17:20])
        self.rf45 = nn.Sequential(*list(original_model.features.children())[21:24])
        self.rf61 = nn.Sequential(*list(original_model.features.children())[24:27])
        self.rf93 = nn.Sequential(*list(original_model.features.children())[28:31])
        self.rf125 = nn.Sequential(*list(original_model.features.children())[31:34])
        self.rf189 = nn.Sequential(*list(original_model.features.children())[35:38])
        self.rf253 = nn.Sequential(*list(original_model.features.children())[38:41])
        self.rf381 = nn.Sequential(*list(original_model.features.children())[42:45])
        self.rf509 = nn.Sequential(*list(original_model.features.children())[45:48])
        self.rf637 = nn.Sequential(*list(original_model.classifier.children())[:-1])
        self.lc = nn.Sequential(*list(original_model.output.children())[:-1])
             
    def forward(self, x):
        x = self.rf3(x)
        rf3 = x
        x = self.rf5(x)
        rf5_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf9(x)
        rf9 = x
        x = self.rf13(x)
        rf13_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf21(x)
        rf21 = x
        x = self.rf29(x)
        rf29_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf45(x)
        rf45 = x
        x = self.rf61(x)
        rf61_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf93(x)
        rf93 = x
        x = self.rf125(x)
        rf125_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf189(x)
        rf189 = x
        x = self.rf253(x)
        rf253_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf381(x)
        rf381 = x
        x = self.rf509(x)
        rf509_p = x
        x = F.max_pool2d(x, 2, 2)
        x = self.rf637(x)
        rf637 = x
        x = F.avg_pool2d(x, 4, 4)
        x = x.view(x.size(0), -1)
        features = x
        x = self.lc(x)
               
        layer_vals = [input, rf3, rf5_p, rf9, rf13_p, rf21, rf29_p, rf45, \
                      rf61_p, rf93, rf125_p, rf189, rf253_p, rf381, rf509_p, \
                      rf637, features, x]
        return layer_vals
