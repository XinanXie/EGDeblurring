import os
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
import torch.nn.functional as f

from pytorch_wavelets import DWTForward, DWTInverse

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm_channel(nn.Module):
    def __init__(self, channel):
        super(LayerNorm_channel, self).__init__()
        self.norm=nn.LayerNorm(channel,eps=1e-5,elementwise_affine=True)

    def forward(self,x):
        h,w=x.shape[-2:]
        x=to_3d(x)
        x=self.norm(x)
        x=to_4d(x,h,w)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, input):
        x = self.fc1(input)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)   #change
        x = self.drop(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, inchannel=128):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inchannel),
            nn.ReLU(),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.ReLU()
            )


    def forward(self, x):

        out = self.block(x)+x

        return out



class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(UpsampleConvLayer, self).__init__()


        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out = self.activation(out)

        return out



####################################################################################################





class EventImage_Fusion_Block_adaDouble(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_Fusion_Block_adaDouble,self).__init__()


        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)
        self.LayerNorm_image_1=LayerNorm_channel(channel=channel)
        self.cnn_img = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.cnn_event = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )   

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.conv_k = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.conv_v0 = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.conv_v1 = nn.Conv2d(channel, channel, kernel_size=1, bias=True)


        self.cnn_0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        

        self.norm2 = nn.LayerNorm(channel)
        mlp_hidden_dim = int(channel * ffn_expansion_factor)
        self.ffn = Mlp(in_features=channel, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)


        

    def forward(self,image,event):
        assert image.shape == event.shape, 'the shape of image does not equal to event'
        b, c , h, w = image.shape
        image_=self.cnn_img(self.LayerNorm_image(image))
        event_=self.cnn_event(self.LayerNorm_event(event))


        q = self.conv_q(image_) 
        k = self.conv_k(event_) 
        v0 = self.conv_v0(event_) 
        v1 = self.conv_v1(event_) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  

        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)

        out0 = (attn @ v0)
        out0 = rearrange(out0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out0 = self.cnn_0(out0)

        out1 = (attn @ v1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.cnn_1(out1)

        out = self.LayerNorm_image_1(image)*out0 + out1

        # mlp
        mod = to_3d(out) # b, h*w, c
        mod = mod + self.ffn(self.norm2(mod))
        mod = to_4d(mod, h, w)


        return mod


class FusionBlock_0(nn.Module):
    def __init__(self,channel,num_heads):
        super(FusionBlock_0,self).__init__()

        self.f0 = EventImage_Fusion_Block_adaDouble(channel,num_heads)
        self.f1 = EventImage_Fusion_Block_adaDouble(channel,num_heads)
        
        
    def forward(self,img,event):

        x2 = self.f0(img,event)
        x2 = self.f1(x2,event)

        return x2




##########################################################################################################################

class Event_Encoder(nn.Module):

    def __init__(self,in_c=6,num_feature_channel=128):
        super(Event_Encoder,self).__init__()

        self.name = 'Event_Encoder'
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
        # self.modulate0=Fourier_attenion_patch(channel=32,patch_size=patch_size)                        
        # self.modulate0=Fourier_attenion_conv(channel=channel_lv0)
        
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
        # self.modulate1=Fourier_attenion_patch(channel=64,patch_size=patch_size)
        # self.modulate1=Fourier_attenion_conv(channel=channel_lv1)
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
        # self.modulate2=Fourier_attenion_patch(channel=128,patch_size=patch_size)
        # self.modulate2=Fourier_attenion_conv(channel=num_feature_channel)
        
    def forward(self, input):

        encoder_feature_list=[]
        #Conv1
        x = self.layer1(input)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        # x = self.modulate0(x) + x
        encoder_feature_list.append(x)
        #Conv2
        x = self.layer4(x)
        x = self.layer5(x) + x
        x = self.layer6(x) + x
        # x = self.modulate1(x) + x
        encoder_feature_list.append(x)
        #Conv3
        x = self.layer7(x)    
        x = self.layer8(x) + x
        x = self.layer9(x) + x 
        # x = self.modulate2(x) + x
        encoder_feature_list.append(x)
        return x,encoder_feature_list



class WLFusionNet(nn.Module):
    def __init__(self,num_input_channel,num_feature_channel=128):
        super(WLFusionNet,self).__init__()

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        self.name = 'WLFusionNet'
        print('build', self.name)


        #encoder
        #Conv1
        self.head0 = nn.Conv2d(num_input_channel, channel_lv0, kernel_size=3, padding=1)
        self.res0 = ResBlock(channel_lv0)
        self.fusion0 = FusionBlock_0(channel_lv0,num_heads=4)
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
        self.fusion1 = FusionBlock_0(channel_lv1,num_heads=4)
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
        self.fusion2 = FusionBlock_0(num_feature_channel,num_heads=8)
        # self.modulate2=Fourier_attenion_conv(channel=num_feature_channel)

        self.resblock0 = ResBlock(num_feature_channel)
        self.resblock1 = ResBlock(num_feature_channel)
        self.resblock2 = ResBlock(num_feature_channel)

        #decoder
        # self.modulate3=Fourier_attenion_conv(channel=num_feature_channel)
        # self.fusion3 = FusionBlock_0(num_feature_channel,num_heads=4)
        self.res30 = ResBlock(num_feature_channel)
        self.res3 = ResBlock(num_feature_channel)
        self.up3 = UpsampleConvLayer(num_feature_channel,channel_lv1,kernel_size=3,padding=1)

        # self.modulate4=Fourier_attenion_conv(channel=channel_lv1)
        # self.fusion4 = FusionBlock_0(channel_lv1,num_heads=4)
        self.res40 = ResBlock(channel_lv1)
        self.res4 = ResBlock(channel_lv1)
        self.up4 = UpsampleConvLayer(channel_lv1,channel_lv0,kernel_size=3,padding=1)

        # self.modulate5=Fourier_attenion_conv(channel=channel_lv0)
        # self.fusion5 = FusionBlock_0(channel_lv0,num_heads=2)
        self.res50 = ResBlock(channel_lv0)
        self.res5 = ResBlock(channel_lv0)
        

        self.tail = nn.Conv2d(channel_lv0,num_input_channel,kernel_size=1)
        self.act = nn.Sigmoid()


    def forward(self,img,eventlist):

        encoder_feature_list = []
        lv1 = f.interpolate(img,scale_factor=1/2,mode='bilinear',align_corners=False)
        lv2 = f.interpolate(img,scale_factor=1/4,mode='bilinear',align_corners=False)

        x = self.head0(img)
        x = self.res0(x)
        x = self.fusion0(x,eventlist[0])
        # x = self.modulate0(x)
        encoder_feature_list.append(x)

        lv1_=self.head1(lv1)
        x = self.down1(x)
        x = x+lv1_
        x = self.res1(x)
        x = self.fusion1(x,eventlist[1])
        # x = self.modulate1(x)
        encoder_feature_list.append(x)

        lv2_=self.head2(lv2)
        x = self.down2(x)
        x = x+lv2_
        x = self.res2(x)
        x = self.fusion2(x,eventlist[2])
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
        
        out = self.act(self.tail(x))

        return out



class WLNet(nn.Module):
    def __init__(self,num_input_channel=3,num_feature_channel=128,num_event_channel=6):
        super(WLNet,self).__init__()

        self.event_encoder = Event_Encoder(in_c=num_event_channel,num_feature_channel=num_feature_channel)
        self.fusionnet = WLFusionNet(num_input_channel,num_feature_channel)


    def forward(self,img,event):
        _,eventlist = self.event_encoder(event)
        out = self.fusionnet(img,eventlist)

        return out



class Event_Interface(nn.Module):
    def __init__(self,num_input_channel=6,num_feature_channel=128):
        super(Event_Interface,self).__init__()

        self.name = 'Event_Interface'
        print('build', self.name)

        channel_lv0 = num_feature_channel//4
        channel_lv1 = num_feature_channel//2

        self.head = nn.Sequential(
            nn.Conv2d(num_input_channel, channel_lv0, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv0, channel_lv1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, num_feature_channel, kernel_size=1),

            )

        self.resblock0 = ResBlock(num_feature_channel)
        self.resblock1 = ResBlock(num_feature_channel)
        self.resblock2 = ResBlock(num_feature_channel)


        self.tail = nn.Sequential(
            nn.Conv2d(num_feature_channel, channel_lv1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv1, channel_lv0, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_lv0, num_input_channel, kernel_size=1)
            )

        self.act = nn.Tanh()


    def forward(self,event):

        x = self.head(event)
        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        out = self.act(self.tail(x))

