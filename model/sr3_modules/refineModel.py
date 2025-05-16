import os
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
import torch.nn.functional as f

from model.sr3_modules.refineModel_utils import DWTfusion_cross,EventImage_Fusion_Block_adaDouble_1, ResBlock, UpsampleLayer



class WLFusionDecoder(nn.Module):
    def __init__(self,num_input_channel,num_feature_channel=128):
        super(WLFusionDecoder,self).__init__()

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        self.name = 'WLFusionDecoder'
        print('build', self.name)


        self.resblock0 = ResBlock(num_feature_channel)
        self.resblock1 = ResBlock(num_feature_channel)
        self.resblock2 = ResBlock(num_feature_channel)

        #decoder
        self.res30 = ResBlock(num_feature_channel)
        self.res3 = ResBlock(num_feature_channel)
        self.up3 = UpsampleLayer(num_feature_channel,channel_lv1,kernel_size=3,padding=1)


        self.res40 = ResBlock(channel_lv1)
        self.res4 = ResBlock(channel_lv1)
        self.up4 = UpsampleLayer(channel_lv1,channel_lv0,kernel_size=3,padding=1)

        self.res50 = ResBlock(channel_lv0)
        self.res5 = ResBlock(channel_lv0)
        

        self.tail = nn.Conv2d(channel_lv0,num_input_channel,kernel_size=1)
        # self.act = nn.Sigmoid()


    def forward(self,x,encoder_feature_list):

        

        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock2(x)

        x = x + encoder_feature_list[2]
        x = self.res30(x)
        x = self.res3(x)
        x = self.up3(x)

        x = x + encoder_feature_list[1]
        x = self.res40(x)
        x = self.res4(x)
        x = self.up4(x)

        x = x + encoder_feature_list[0]
        x = self.res50(x)
        x = self.res5(x)
        
        # out = self.act(self.tail(x))
        out = self.tail(x)

        return out

class EventEncoderFusion(nn.Module):

    def __init__(self,in_c=6,num_feature_channel=128):
        super(EventEncoderFusion,self).__init__()

        self.name = 'EventEncoderFusion'
        print('build', self.name)

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        #Conv1
        self.layer1 = nn.Conv2d(in_c, channel_lv0, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel_lv0, channel_lv0, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv0),
            nn.ReLU(),
            nn.Conv2d(channel_lv0, channel_lv0, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv0)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(channel_lv0, channel_lv0, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv0),
            nn.ReLU(),
            nn.Conv2d(channel_lv0, channel_lv0, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv0)
            )
        self.fusion0 = FusionBlock_0(channel_lv0,num_heads=4)
        
        
        #Conv2
        self.layer4 = nn.Conv2d(channel_lv0, channel_lv1, kernel_size=3, stride=2, padding=1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv1)
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channel_lv1)
            )
        self.fusion1 = FusionBlock_0(channel_lv1,num_heads=4)
        
        #Conv3
        self.layer7 = nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=3, stride=2, padding=1)
        self.layer8 = nn.Sequential(
            nn.Conv2d(num_feature_channel, num_feature_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_feature_channel),
            nn.ReLU(),
            nn.Conv2d(num_feature_channel, num_feature_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_feature_channel)
            )
        self.layer9 = nn.Sequential(
            nn.Conv2d(num_feature_channel, num_feature_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_feature_channel),
            nn.ReLU(),
            nn.Conv2d(num_feature_channel, num_feature_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_feature_channel),
            )
        self.fusion2 = FusionBlock_0(num_feature_channel,num_heads=8)
        
    def forward(self,imagelist,event):

        encoder_feature_list=[]
        #Conv1
        x = self.layer1(event)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.fusion0(imagelist[0],x)
        encoder_feature_list.append(x)
        #Conv2
        x = self.layer4(x)
        x = self.layer5(x) + x
        x = self.layer6(x) + x
        x = self.fusion1(imagelist[1],x)
        encoder_feature_list.append(x)
        #Conv3
        x = self.layer7(x)    
        x = self.layer8(x) + x
        x = self.layer9(x) + x 
        x = self.fusion2(imagelist[2],x)
        encoder_feature_list.append(x)
        return x,encoder_feature_list





class FusionBlock_0(nn.Module):
    def __init__(self,channel,num_heads):
        super(FusionBlock_0,self).__init__()
        print('FusionBlock_cross')
        
        self.f0 = EventImage_Fusion_Block_adaDouble_1(channel,num_heads)
        self.fx = DWTfusion_cross(channel,num_heads)
        
    def forward(self,img,event):

        x2 = self.fx(img,event)
        x2 = self.f0(x2,event)

        return x2



class ImageEncoder(nn.Module):
    def __init__(self,num_input_channel,num_feature_channel=128):
        super(ImageEncoder,self).__init__()

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        self.name = 'ImageEncoder'
        print('build', self.name)


        #encoder
        #Conv1
        self.head0 = nn.Conv2d(num_input_channel, channel_lv0, kernel_size=3, padding=1)
        self.res0 = ResBlock(channel_lv0)
        self.res00 = ResBlock(channel_lv0)
        
        #Conv2
        self.head1 = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Conv2d(channel_lv0, channel_lv1, kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock(channel_lv1)
        self.res11 = ResBlock(channel_lv1)
        
        #Conv3
        self.head2 = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(num_feature_channel)
        self.res22 = ResBlock(num_feature_channel)
        

    def forward(self,img):

        encoder_feature_list = []
        lv1 = f.interpolate(img,scale_factor=1/2,mode='bilinear',align_corners=False)
        lv2 = f.interpolate(img,scale_factor=1/4,mode='bilinear',align_corners=False)

        x = self.head0(img)
        x = self.res0(x)
        x = self.res00(x)
        encoder_feature_list.append(x)

        lv1_=self.head1(lv1)
        x = self.down1(x)
        x = x+lv1_
        x = self.res1(x)
        x = self.res11(x)
        encoder_feature_list.append(x)

        lv2_=self.head2(lv2)
        x = self.down2(x)
        x = x+lv2_
        x = self.res2(x)
        x = self.res22(x)
        encoder_feature_list.append(x)

        

        return x,encoder_feature_list




#基于细节丰富的融合网络
class RDBlock_1(nn.Module):
    def __init__(self,channel,num_heads):
        super(RDBlock_1,self).__init__()
        
        self.f0 = EventImage_Fusion_Block_adaDouble_1(channel,num_heads)

        self.Upsample= nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        
        
    def forward(self,img,LR_feature):

        SR_feature = self.Upsample(LR_feature)

        x = self.f0(img,SR_feature)
        
        return x



class SRUnet(nn.Module):
    def __init__(self,num_input_channel,num_feature_channel=128):
        super(SRUnet,self).__init__()

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        self.name = 'SRUnet'
        print('build', self.name)


        #encoder
        #Conv1
        self.head0 = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv0, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv0, channel_lv0, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.res0 = ResBlock(channel_lv0)
        self.fusion0 = RDBlock_1(channel_lv0,num_heads=4)
        # self.modulate0=Fourier_attenion_conv(channel=channel_lv0)
        
        #Conv2
        self.head1 = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Conv2d(channel_lv0, channel_lv1, kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock(channel_lv1)
        self.fusion1 = RDBlock_1(channel_lv1,num_heads=4)
        # self.modulate1=Fourier_attenion_conv(channel=channel_lv1)
        #Conv3
        self.head2 = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(num_feature_channel)
        self.fusion2 = RDBlock_1(num_feature_channel,num_heads=8)
        # self.modulate2=Fourier_attenion_conv(channel=num_feature_channel)

        self.resblock0 = ResBlock(num_feature_channel)
        self.resblock1 = ResBlock(num_feature_channel)
        self.resblock2 = ResBlock(num_feature_channel)

        #decoder
        # self.modulate3=Fourier_attenion_conv(channel=num_feature_channel)
        # self.fusion3 = FusionBlock_0(num_feature_channel,num_heads=4)
        self.res30 = ResBlock(num_feature_channel)
        self.res3 = ResBlock(num_feature_channel)
        self.up3 = UpsampleLayer(num_feature_channel,channel_lv1,kernel_size=3,padding=1)

        # self.modulate4=Fourier_attenion_conv(channel=channel_lv1)
        # self.fusion4 = FusionBlock_0(channel_lv1,num_heads=4)
        self.res40 = ResBlock(channel_lv1)
        self.res4 = ResBlock(channel_lv1)
        self.up4 = UpsampleLayer(channel_lv1,channel_lv0,kernel_size=3,padding=1)

        # self.modulate5=Fourier_attenion_conv(channel=channel_lv0)
        # self.fusion5 = FusionBlock_0(channel_lv0,num_heads=2)
        self.res50 = ResBlock(channel_lv0)
        self.res5 = ResBlock(channel_lv0)
        

        self.tail = nn.Conv2d(channel_lv0,num_input_channel,kernel_size=1)
        # self.act = nn.Sigmoid()


    def forward(self,img,LRlist):

        encoder_feature_list = []
        lv0 = f.interpolate(img,scale_factor=4,mode='bilinear',align_corners=False)
        lv1 = f.interpolate(img,scale_factor=2,mode='bilinear',align_corners=False)
        lv2 = img

        x = self.head0(lv0)
        x = self.res0(x)
        x = self.fusion0(x,LRlist[0])
        # x = self.modulate0(x)
        encoder_feature_list.append(x)

        lv1_=self.head1(lv1)
        x = self.down1(x)
        x = x+lv1_
        x = self.res1(x)
        x = self.fusion1(x,LRlist[1])
        # x = self.modulate1(x)
        encoder_feature_list.append(x)

        lv2_=self.head2(lv2)
        x = self.down2(x)
        x = x+lv2_
        x = self.res2(x)
        x = self.fusion2(x,LRlist[2])
        # x = self.modulate2(x)
        encoder_feature_list.append(x)

        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock2(x)

        x = x + encoder_feature_list[2]
        # x = self.modulate3(x)
        # x = self.fusion3(x,eventlist[2])
        x = self.res30(x)
        x = self.res3(x)
        x = self.up3(x)

        x = x + encoder_feature_list[1]
        # x = self.modulate4(x)
        # x = self.fusion4(x,eventlist[1])
        x = self.res40(x)
        x = self.res4(x)
        x = self.up4(x)

        x = x + encoder_feature_list[0]
        # x = self.modulate5(x)
        # x = self.fusion5(x,eventlist[0])
        x = self.res50(x)
        x = self.res5(x)
        
        # out = self.act(self.tail(x))
        out = self.tail(x)

        return out


#################################################################################################



#添加跨分量
class RDSRNet_h(nn.Module):
    def __init__(self,num_input_channel=3,num_feature_channel=128,num_event_channel=6):
        super(RDSRNet_h,self).__init__()

        self.event_encoder = EventEncoderFusion(in_c=num_event_channel,num_feature_channel=num_feature_channel)
        self.image_encoder = ImageEncoder(num_input_channel,num_feature_channel)
        self.decoder = WLFusionDecoder(num_input_channel,num_feature_channel)
        self.srnet = SRUnet(num_input_channel,num_feature_channel)


    def forward(self,img,event):
        
        _,imagelist = self.image_encoder(img)
        x,LRlist = self.event_encoder(imagelist,event)
        sharp_LR = self.decoder(x,LRlist)
        out =  self.srnet(img,LRlist)

        return out,sharp_LR