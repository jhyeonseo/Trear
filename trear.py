import resnet
import torch.nn as nn
import torch
import torchvision
from PIL import Image, ImageOps
from torch.nn.init import normal_ as normal
from torch.nn.init import constant_ as constant

from typing import Optional
from torch import Tensor
import random
from torch.autograd import Variable

from transformer import Transformer_Encoder, PositionalEncoding, Transformer_CA

class ConsensusModule(torch.nn.Module):
    def __init__(self, dim=1):
        super(ConsensusModule, self).__init__()
        self.dim = dim

    def forward(self, input):
        output = input.mean(dim=self.dim, keepdim=True)

        return output

class FusionNet(nn.Module):
    def __init__(self, backbone_dim=2048, c_dim=512, num_c=45, num_segments=32):
        super(FusionNet, self).__init__()
        self.c_dim = c_dim  # channel number after reducing the dimension
        self.backbone_dim = backbone_dim
        self.droprate = 0.5  # transformer's droprate
        self.nheads = 8
        self.dim_feedforward = 2048  # the Number of hidden nodes in the intermediate MLP
        self.layers = 1
        self.num_segments = num_segments
        
        self.pos_rgb = PositionalEncoding(c_dim)
        self.pos_depth = PositionalEncoding(c_dim)
        self.transformer_rgb=Transformer_Encoder(
                        d_model=c_dim,
                        nhead=self.nheads, 
                        num_encoder_layers=self.layers,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.droprate,
                        activation="gelu")
        self.transformer_depth=Transformer_Encoder(
                        d_model=c_dim,
                        nhead=self.nheads, 
                        num_encoder_layers=self.layers,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.droprate,
                        activation="gelu")
        self.transformer_fusion=Transformer_CA(
                        d_model=c_dim,
                        nhead=self.nheads, 
                        num_encoder_layers=1,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.droprate,
                        activation="gelu")
        

        self.reduce_channel1 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=True)
        self.reduce_channel2 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(c_dim)
        self.bn2 = nn.BatchNorm2d(c_dim)
        self.avgpool1 = nn.AdaptiveAvgPool1d(output_size=c_dim)
        self.avgpool2 = nn.AdaptiveAvgPool1d(output_size=c_dim)
        self.fc_out = nn.Linear(c_dim, num_c)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, rgb_feat, depth_feat):
        # Channel Reduction
        rgb_feat = self.relu1(self.reduce_channel1(rgb_feat))
        rgb_feat = self.bn1(rgb_feat)
        rgb_feat = self.dropout(rgb_feat)
        depth_feat = self.relu2(self.reduce_channel2(depth_feat))
        depth_feat = self.bn2(depth_feat)
        depth_feat = self.dropout(depth_feat)
        
        # Conduct average pooling on the fea-ture maps of each frames to produce the feature embeddings{F1r , F2r , ..., Fkr } with size dmodel = 512.
        # (N, L, E),where L is the target sequence length, N is the batch size, E is the embedding dimension. 
        ############### Average pooling #################
        rgb_feat = rgb_feat.flatten(1)
        rgb_feat = rgb_feat.view(-1,self.num_segments,rgb_feat.size(1)).contiguous()
        depth_feat = depth_feat.flatten(1)
        depth_feat = depth_feat.view(-1,self.num_segments,depth_feat.size(1)).contiguous()
        rgb_feat = self.avgpool1(rgb_feat)
        depth_feat = self.avgpool2(depth_feat)


        ######################  SELF ATTENTION ####################
        rgb_feat_pos = self.pos_rgb(rgb_feat)
        depth_feat_pos = self.pos_depth(depth_feat)
    
        rgb_sa, _ = self.transformer_rgb(src=rgb_feat, src_pos=rgb_feat_pos)
        depth_sa, _ = self.transformer_depth(src=depth_feat, src_pos=depth_feat_pos)
        ######################  CROSS ATTENTION ####################
        feat_fus, _ = self.transformer_fusion(src1=rgb_sa, src2=depth_sa)
        
        ######################  Classification ####################
        feat_fus = self.fc_out(feat_fus)
        feat_fus = self.dropout(feat_fus)

       
        return feat_fus



class Trear(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet34', new_length=None,
                 dropout=0.5,
                 crop_num=1):
        super(Trear, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.dropout = dropout
        self.crop_num = crop_num
        self.arch = base_model
        
          
        if new_length is None:
            self.new_length = 1 
        else:
            self.new_length = new_length

        print(("""
        Initializing Trear with base model: {}.
        Trear Configurations:
        num_segments:       {}
        new_length:         {}
        dropout_ratio:      {}
        """.format(base_model, self.num_segments, self.new_length, self.dropout)))
        
        self._prepare_base_model(base_model)
        self.consensus = ConsensusModule()

            
        if base_model == 'resnet101' or base_model == 'resnet50':
            self.fusmodel = FusionNet(backbone_dim=2048, c_dim=512, num_c=num_class, num_segments=self.num_segments)
        elif base_model == 'resnet34' or base_model == 'resnet18':
            self.fusmodel = FusionNet(backbone_dim=512, c_dim=512, num_c=num_class, num_segments=self.num_segments)

    def _prepare_base_model(self, base_model):
        if base_model == 'resnet101':
            self.base_model = resnet.resnet101(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
        elif base_model == 'resnet50':
            self.base_model = resnet.resnet50(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
        elif base_model == 'resnet34':
            self.base_model = resnet.resnet34(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        super(Trear, self).train(mode)
        #'''
        for (name, param) in self.base_model.named_parameters():    
            if 'layer' in name:
                param.requires_grad = False
        #'''

    def forward(self, input):
        sample_len = 6 * self.new_length  # Connect two images by channel 6

        '''
        img = input.view((-1, sample_len) + input.size()[-2:]).clone().detach().to('cpu')    
        print(img.shape)
        for k in range(int(img.shape[0])):
            for i in range(int(img.shape[1]/3)):
                r = img[k][3*i].clone() * 0.226 + 0.5
                r = r.unsqueeze(0)
                g = img[k][3*i+1].clone()* 0.226 + 0.5
                g = g.unsqueeze(0)
                b = img[k][3*i+2].clone()* 0.226 + 0.5
                b = b.unsqueeze(0)
                
                full = torch.cat((r,g,b),dim=0)
                plt.imshow(full.permute(1, 2, 0))
                plt.show()         
        '''
        
        base_out_1, base_out_2 = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        fus_out = self.fusmodel(base_out_1, base_out_2)
        output = self.consensus(fus_out)
        
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]), GroupRandomHorizontalFlip(is_flow=False)])
    
    
 ################## Augumentation  ##################  
class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        ret_img_group = [img.resize((256, 256)) for img in img_group]
        crop_img_group = []
        for img in ret_img_group:
            random_number_x = random.randint(0, 255 - self.input_size[0])
            random_number_y = random.randint(0, 255 - self.input_size[1])
            crop_img_group.append(img.crop((random_number_x, random_number_y, random_number_x + self.input_size[0], random_number_y + self.input_size[1])))
            #print(random_number_x, random_number_y, random_number_x + self.input_size[0], random_number_y + self.input_size[1])

        '''    
        for img in ret_img_group:
            plt.imshow(img)
            plt.show()
        '''
        return crop_img_group
    
    
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):        
        v = random.random()
            
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            '''
            for img in ret:
                plt.imshow(img)
                plt.show()
            '''
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group