import pytorchvideo.models.resnet as resnet 
import pytorchvideo.models.slowfast as slowfast 

import torch.nn as nn 

def make_kinetics_resnet(): 
    resnet_model = resnet.create_resnet(
        input_channel = 3, 
        model_depth = 50, 
        model_num_class = 3, 
        norm = nn.BatchNorm3d, 
        activation = nn.ReLU 
    )

    return resnet_model 

def make_slowfast(): 
    slowfast_model = slowfast.create_slowfast() 

    return slowfast_model 