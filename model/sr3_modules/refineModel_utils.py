
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


#采用自注意力在频率域挑选事件
class Fourier_attenion_conv(nn.Module):
    def __init__(self,channel,hidden_channel_factor=2):
        super(Fourier_attenion_conv,self).__init__()

        hidden_channel=int(channel*hidden_channel_factor)


        self.LN=LayerNorm_channel(channel)
        #频域
        self.conv_f1=nn.Conv2d(channel,hidden_channel*2,kernel_size=1)
        self.conv_f2=nn.Conv2d(hidden_channel*2,hidden_channel*2,kernel_size=1)
        self.conv_f_ri = nn.Sequential(
            nn.Conv2d(hidden_channel*4, hidden_channel*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*4, hidden_channel*4, kernel_size=1),
            nn.Sigmoid()
            )
        

        self.out=nn.Conv2d(hidden_channel,channel,kernel_size=1)


    def forward(self,x):

        x_ln=self.LN(x)

        #频域
        x_f=self.conv_f1(x_ln)
        
        b,c,h,w=x_f.shape
        x_f=torch.fft.rfft2(x_f)
        x_real_imag=torch.cat([x_f.real,x_f.imag],dim=1)
        atten = self.conv_f_ri(x_real_imag)
        x_real_imag = atten * x_real_imag
        x_real,x_imag = torch.chunk(x_real_imag,2,dim=1)
        x_f = torch.complex(x_real,x_imag)
        x_f=torch.fft.irfft2(x_f,s=(h, w))
        
        x_f1,x_f2=self.conv_f2(x_f).chunk(2, dim=1)
        x_f = f.gelu(x_f1) * x_f2

        x_f = self.out(x_f)


        return x_f



#采用自注意力在频率域选择特征频道
class Fourier_attenion_channel(nn.Module):
    def __init__(self,channel,hidden_channel_factor=2):
        super(Fourier_attenion_channel,self).__init__()

        hidden_channel=int(channel*hidden_channel_factor)


        self.LN=LayerNorm_channel(channel)
        #频域
        self.conv_f1=nn.Conv2d(channel,hidden_channel*2,kernel_size=1)
        self.conv_f2=nn.Conv2d(hidden_channel*2,hidden_channel*2,kernel_size=1)
        self.conv_f_ri = nn.Sequential(
            nn.Conv2d(hidden_channel*4, hidden_channel*4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*4, hidden_channel*4, kernel_size=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(hidden_channel*4, hidden_channel*3, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*3, hidden_channel*4, 1),
            nn.Sigmoid()
            )
        

        self.out=nn.Conv2d(hidden_channel,channel,kernel_size=1)


    def forward(self,x):

        x_ln=self.LN(x)

        #频域
        x_f=self.conv_f1(x_ln)
        
        b,c,h,w=x_f.shape
        x_f=torch.fft.rfft2(x_f)
        x_real_imag=torch.cat([x_f.real,x_f.imag],dim=1)
        atten = self.conv_f_ri(x_real_imag)
        x_real_imag = atten * x_real_imag
        x_real,x_imag = torch.chunk(x_real_imag,2,dim=1)
        x_f = torch.complex(x_real,x_imag)
        x_f=torch.fft.irfft2(x_f,s=(h, w))
        
        x_f1,x_f2=self.conv_f2(x_f).chunk(2, dim=1)
        x_f = f.gelu(x_f1) * x_f2
        
        #空域
        # x_s=self.conv_s1(x_ln)
        # x_s1,x_s2=x_s.chunk(2, dim=1)
        # x_s = torch.cat([self.norm(x_s1), x_s2], dim=1)
        # x_s=self.relu_s1(x_s)
        # x_s=self.relu_s2(self.conv_s2(x_s))+self.connect(x_ln)


        x_f = self.out(x_f)


        return x_f

#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention_Block(nn.Module):
    def __init__(self, planes):
        super(ChannelAttention_Block, self).__init__()
        self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        # x = self.sa(x) * x
        return x

class EventImage_Fusion_Block_0(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_Fusion_Block_0,self).__init__()

        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)
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
        self.conv_v = nn.Conv2d(channel, channel, kernel_size=1, bias=True)


        self.cnn_0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
            )

        # self.cnn_2 = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        #     )

    def forward(self,image,event):
        assert image.shape == event.shape, 'the shape of image does not equal to event'
        b, c , h, w = image.shape
        image_=self.cnn_img(self.LayerNorm_image(image))
        event_=self.cnn_event(self.LayerNorm_event(event))

        q = self.conv_q(image_) 
        k = self.conv_k(image_) 
        v = self.conv_v(event_) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  

        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        mod = (attn @ v)
        mod = rearrange(mod, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        image_ = self.cnn_0(image_)
        out = self.cnn_1(image_*mod)

        return out

#在原有基础上添加新内容 （2023.11.23）
class EventImage_Fusion_Block_1(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_Fusion_Block_1,self).__init__()


        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)
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
        self.conv_v = nn.Conv2d(channel, channel, kernel_size=1, bias=True)


        self.cnn_0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        

        self.norm2 = nn.LayerNorm(channel)
        mlp_hidden_dim = int(channel * ffn_expansion_factor)
        self.ffn = Mlp(in_features=channel, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)



        #直接链接学习

        # self.cnn_1 = nn.Sequential(
        #     nn.Conv2d(channel*2, channel*2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(channel*2, channel, kernel_size=1, padding=0)
        #     )

        

    def forward(self,image,event):
        assert image.shape == event.shape, 'the shape of image does not equal to event'
        b, c , h, w = image.shape
        image_=self.cnn_img(self.LayerNorm_image(image))
        event_=self.cnn_event(self.LayerNorm_event(event))

        q = self.conv_q(image_) 
        k = self.conv_k(image_) 
        v = self.conv_v(event_) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  

        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # x = torch.cat((image_,event_),dim=1)
        # x = self.cnn_1(x) * image_

        out = self.cnn_0(out) + image_

        # mlp
        mod = to_3d(out) # b, h*w, c
        mod = mod + self.ffn(self.norm2(mod))
        mod = to_4d(mod, h, w)


        return mod






#在原有基础上修改v （2023.12.14）
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



#添加prefusion（2024.03.14）
class EventImage_Fusion_Block_adaDouble_1(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_Fusion_Block_adaDouble_1,self).__init__()


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

        self.conv_exchange = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*2, channel, kernel_size=1, padding=0)
            )   

        self.pre_fusion = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
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

        x_exchange = self.conv_exchange(torch.cat((image_,event_),dim=1))
        event_ = self.pre_fusion(torch.cat((event_,x_exchange),dim=1))

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


class DWT_Upsample_Fusion_x4(nn.Module):
    def __init__(self, channel,num_heads=4):
        super(DWT_Upsample_Fusion_x4,self).__init__()
        self.dwt_img = DWTForward(J=2, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')

        self.conv_LH_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HL_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HH_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.conv_LH_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HL_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HH_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.upsample=nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        # self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

    def forward(self,fb,fe):
        fb_LL,fb_H = self.dwt_img(fb)
        fb_H_lv1 = fb_H[0]
        fb_LH_lv1,fb_HL_lv1,fb_HH_lv1 = torch.chunk(fb_H_lv1,3,dim=2)
        fb_LH_lv1 = fb_LH_lv1.squeeze(dim = 2)
        fb_HL_lv1 = fb_HL_lv1.squeeze(dim = 2)
        fb_HH_lv1 = fb_HH_lv1.squeeze(dim = 2)

        fb_H_lv2 = fb_H[1]
        fb_LH_lv2,fb_HL_lv2,fb_HH_lv2 = torch.chunk(fb_H_lv2,3,dim=2)
        fb_LH_lv2 = fb_LH_lv2.squeeze(dim = 2)
        fb_HL_lv2 = fb_HL_lv2.squeeze(dim = 2)
        fb_HH_lv2 = fb_HH_lv2.squeeze(dim = 2)

        #处理低频
        fh_LL = self.conv_LL(torch.cat((fb_LL,fe),dim=1))
        #处理lv2
        fh_LH_lv2 = self.conv_LH_lv2(torch.cat((fb_LH_lv2,fe),dim=1))
        fh_HL_lv2 = self.conv_HL_lv2(torch.cat((fb_HL_lv2,fe),dim=1))
        fh_HH_lv2 = self.conv_HH_lv2(torch.cat((fb_HH_lv2,fe),dim=1))

        #处理lv1
        fe_up = self.upsample(fe)
        fh_LH_lv1 = self.conv_LH_lv1(torch.cat((fb_LH_lv1,fe_up),dim=1))
        fh_HL_lv1 = self.conv_HL_lv1(torch.cat((fb_HL_lv1,fe_up),dim=1))
        fh_HH_lv1 = self.conv_HH_lv1(torch.cat((fb_HH_lv1,fe_up),dim=1))

        Wh_lv2 = torch.stack((fh_LH_lv2,fh_HL_lv2,fh_HH_lv2),dim=2) #B C 3 H W
        Wh_lv1 = torch.stack((fh_LH_lv1,fh_HL_lv1,fh_HH_lv1),dim=2) #B C 3 H W

        Whlist = [Wh_lv1,Wh_lv2]

        y = self.iwt_img((fh_LL,Whlist))

        return y

class DWT_Upsample_Fusion_x4_conv1x1(nn.Module):
    def __init__(self, channel,num_heads=4):
        super(DWT_Upsample_Fusion_x4_conv1x1,self).__init__()
        self.dwt_img = DWTForward(J=2, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')

        self.conv_LH_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HL_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HH_lv1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.conv_LH_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HL_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )
        self.conv_HH_lv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        self.upsample=nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

        # self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            )

    def forward(self,fb,fe):
        fb_LL,fb_H = self.dwt_img(fb)
        fb_H_lv1 = fb_H[0]
        fb_LH_lv1,fb_HL_lv1,fb_HH_lv1 = torch.chunk(fb_H_lv1,3,dim=2)
        fb_LH_lv1 = fb_LH_lv1.squeeze(dim = 2)
        fb_HL_lv1 = fb_HL_lv1.squeeze(dim = 2)
        fb_HH_lv1 = fb_HH_lv1.squeeze(dim = 2)

        fb_H_lv2 = fb_H[1]
        fb_LH_lv2,fb_HL_lv2,fb_HH_lv2 = torch.chunk(fb_H_lv2,3,dim=2)
        fb_LH_lv2 = fb_LH_lv2.squeeze(dim = 2)
        fb_HL_lv2 = fb_HL_lv2.squeeze(dim = 2)
        fb_HH_lv2 = fb_HH_lv2.squeeze(dim = 2)

        #处理低频
        fh_LL = self.conv_LL(torch.cat((fb_LL,fe),dim=1))
        #处理lv2
        fh_LH_lv2 = self.conv_LH_lv2(torch.cat((fb_LH_lv2,fe),dim=1))
        fh_HL_lv2 = self.conv_HL_lv2(torch.cat((fb_HL_lv2,fe),dim=1))
        fh_HH_lv2 = self.conv_HH_lv2(torch.cat((fb_HH_lv2,fe),dim=1))

        #处理lv1
        fe_up = self.upsample(fe)
        fh_LH_lv1 = self.conv_LH_lv1(torch.cat((fb_LH_lv1,fe_up),dim=1))
        fh_HL_lv1 = self.conv_HL_lv1(torch.cat((fb_HL_lv1,fe_up),dim=1))
        fh_HH_lv1 = self.conv_HH_lv1(torch.cat((fb_HH_lv1,fe_up),dim=1))

        Wh_lv2 = torch.stack((fh_LH_lv2,fh_HL_lv2,fh_HH_lv2),dim=2) #B C 3 H W
        Wh_lv1 = torch.stack((fh_LH_lv1,fh_HL_lv1,fh_HH_lv1),dim=2) #B C 3 H W

        Whlist = [Wh_lv1,Wh_lv2]

        y = self.iwt_img((fh_LL,Whlist)) + fb

        return y


#渐进式上采样
class DWT_Upsample(nn.Module):
    def __init__(self, channel):
        super(DWT_Upsample,self).__init__()
        # self.dwt_img = DWTForward(J=2, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')

        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
            )

        self.conv_LH = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
            )
        
        self.conv_HL = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
            )
        
        self.conv_HH = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
            )

        self.conv_out = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel//2, channel//2, kernel_size=1)
            )

        self.upsample=nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channel, channel//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel//2, channel//2, kernel_size=1)
            )

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)

    def forward(self,img,guide):
        img_LL = self.conv_LL(torch.cat((img,guide),dim=1))
        img_LH = self.conv_LH(torch.cat((img,guide),dim=1))
        img_HL = self.conv_LL(torch.cat((img,guide),dim=1))
        img_HH = self.conv_LL(torch.cat((img,guide),dim=1))

        Wh = torch.stack((img_LH,img_HL,img_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)
        Whlist = [Wh]
        y = self.iwt_img((img_LL, Whlist))

        img_ = self.upsample(img)

        y = self.conv_out(y)+img_

        return y





class Mutual_Attention(nn.Module):
    def __init__(self, channel, num_heads=1, bias=False):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = nn.Conv2d(channel, channel, kernel_size=1, bias=bias)
        self.conv_k = nn.Conv2d(channel, channel, kernel_size=1, bias=bias)
        self.conv_v = nn.Conv2d(channel, channel, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(channel, channel, kernel_size=1, bias=bias)

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.conv_q(x) # image
        k = self.conv_k(y) # event
        v = self.conv_v(y) # event   #change!!!

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   #change

        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class EventImage_Fusion_Block(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_Fusion_Block,self).__init__()

        self.LayerNorm_event=LayerNorm_channel(channel=channel)   #change
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   #change

        self.attn = Mutual_Attention(channel, num_heads, bias=False)

        self.norm2 = nn.LayerNorm(channel)
        mlp_hidden_dim = int(channel * ffn_expansion_factor)
        self.ffn = Mlp(in_features=channel, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self,image,event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w

        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.LayerNorm_image(image), self.LayerNorm_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused



class EventImage_DWT_Fusion(nn.Module):
    def __init__(self,channel,num_heads,ffn_expansion_factor=2):
        super(EventImage_DWT_Fusion,self).__init__()


        self.attn = Mutual_Attention(channel, num_heads, bias=False)

        self.norm2 = nn.LayerNorm(channel)
        mlp_hidden_dim = int(channel * ffn_expansion_factor)
        self.ffn = Mlp(in_features=channel, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self,image,event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w

        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(image, event) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused



# #不能加InstanceNorm2d
# class DWTfusion(nn.Module):
#     def __init__(self,channel):
#         super(DWTfusion,self).__init__()

#         self.dwt = DWT()
#         self.iwt = IWT()
#         self.LayerNorm_event=LayerNorm_channel(channel=channel)   
#         self.LayerNorm_image=LayerNorm_channel(channel=channel)   

#         self.conv_img_LL = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#         )

#         self.conv_img_H = nn.Sequential(
#             nn.Conv2d(channel*3,channel*3,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*3,channel*3,kernel_size=3,padding=1),
#         )

#         self.conv_event = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#         )

#         self.fusion = nn.Sequential(
#             nn.Conv2d(channel*4,channel*4,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*4,channel*4,kernel_size=1),
#             nn.LeakyReLU()
#         )

#         self.mult = nn.Sequential(
#             nn.Conv2d(channel*4,channel*4,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*4,channel*3,kernel_size=1),
#             nn.Sigmoid()
#         )

#         self.sum = nn.Sequential(
#             nn.Conv2d(channel*4,channel*4,kernel_size=3,padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*4,channel*3,kernel_size=1),
#             nn.LeakyReLU()
#         )




#     def forward(self,img,event):

#         img_LL,img_H = self.dwt(img) #img_H包括其他分量

#         img_LL = self.conv_img_LL(img_LL)
#         img_H = self.conv_img_H(img_H)

#         event_0 = f.interpolate(event, scale_factor=1/2, mode='bilinear', align_corners=False)
#         event_0 = self.conv_event(event_0)

#         fusion_fea = torch.cat((img_H,event_0),dim=1)
#         fusion_fea = self.fusion(fusion_fea)

#         mult_fea = self.mult(fusion_fea)
#         sum_fea = self.sum(fusion_fea)

#         x = img_H*mult_fea + sum_fea
#         y = torch.cat((img_LL,x),dim=1)
#         y = self.iwt(y)

#         return y


# class DWTfusionblock(nn.Module):
#     def __init__(self,channel):
#         super(DWTfusionblock,self).__init__()

#         self.conv_img = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1),
#             # nn.InstanceNorm2d(channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=1)
#         )

#         self.conv_event = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1),
#             # nn.InstanceNorm2d(channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=1)
#         )

#         self.fusion_mult = nn.Sequential(
#             nn.Conv2d(channel*2,channel*2,kernel_size=1),
#             # nn.InstanceNorm2d(channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*2,channel,kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.fusion_sum = nn.Sequential(
#             nn.Conv2d(channel*2,channel*2,kernel_size=1),
#             # nn.InstanceNorm2d(channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel*2,channel,kernel_size=1),
#         )

#     def forward(self,img_dwt,event_dwt):

#         img = self.conv_img(img_dwt)
#         event = self.conv_img(event_dwt)

#         fusion_fea = torch.cat((img,event),dim=1)
#         fusion_fea = self.fusion_mult(fusion_fea)*img + self.fusion_sum(fusion_fea) + img

#         return fusion_fea




# #采用API
# #event取wavelet
# class DWTfusion_2(nn.Module):
#     def __init__(self,channel):
#         super(DWTfusion_2,self).__init__()

#         self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
#         self.iwt_img = DWTInverse(wave='haar', mode='zero')
#         self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
#         self.LayerNorm_event=LayerNorm_channel(channel=channel)   
#         self.LayerNorm_image=LayerNorm_channel(channel=channel)   

#         self.conv_img_LL = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1),
#             # nn.InstanceNorm2d(channel),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=1)
#         )

#         self.LH = DWTfusionblock(channel)
#         self.HL = DWTfusionblock(channel)
#         self.HH = DWTfusionblock(channel)

        
        
#     def forward(self,img_,event_):

#         event = self.LayerNorm_event(event_)
#         img = self.LayerNorm_image(img_)


#         img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

#         img_H_ = img_H[0]
#         img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
#         img_LH = img_LH.squeeze(dim = 2)
#         img_HL = img_HL.squeeze(dim = 2)
#         img_HH = img_HH.squeeze(dim = 2)

#         event_LL,event_H = self.dwt_event(event) #img_H包括其他分量

#         event_H_ = event_H[0]
#         event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
#         event_LH = event_LH.squeeze(dim = 2)
#         event_HL = event_HL.squeeze(dim = 2)
#         event_HH = event_HH.squeeze(dim = 2)

#         img_LL = self.conv_img_LL(img_LL)

#         fusion_fea_LH = self.LH(img_LH,event_LH)
#         fusion_fea_HL = self.HL(img_HL,event_HL)
#         fusion_fea_HH = self.HH(img_HH,event_HH)


#         Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2)
#         Whlist = [Wh]

#         y = self.iwt_img((img_LL, Whlist))

#         return y






# class DWTfusion_3(nn.Module):
#     def __init__(self,channel,num_heads):
#         super(DWTfusion_3,self).__init__()

#         self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
#         self.iwt_img = DWTInverse(wave='haar', mode='zero')
#         self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
#         self.LayerNorm_event=LayerNorm_channel(channel=channel)   
#         self.LayerNorm_image=LayerNorm_channel(channel=channel)   

#         self.conv_img_LL = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=1)
#         )


#         self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
#         self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
#         self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        
        
#     def forward(self,img_,event_):

#         event = self.LayerNorm_event(event_)
#         img = self.LayerNorm_image(img_)


#         img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

#         img_H_ = img_H[0]
#         img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
#         img_LH = img_LH.squeeze(dim = 2)
#         img_HL = img_HL.squeeze(dim = 2)
#         img_HH = img_HH.squeeze(dim = 2)

#         event_LL,event_H = self.dwt_event(event) #img_H包括其他分量

#         event_H_ = event_H[0]
#         event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
#         event_LH = event_LH.squeeze(dim = 2)
#         event_HL = event_HL.squeeze(dim = 2)
#         event_HH = event_HH.squeeze(dim = 2)

#         img_LL = self.conv_img_LL(img_LL)

#         fusion_fea_LH = self.LH(img_LH,event_LH)
#         fusion_fea_HL = self.HL(img_HL,event_HL)
#         fusion_fea_HH = self.HH(img_HH,event_HH)


#         Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2)
#         Whlist = [Wh]

#         y = self.iwt_img((img_LL, Whlist))

#         return y


# #高频部分合起来调制
# class DWTfusion_4(nn.Module):
#     def __init__(self,channel,num_heads):
#         super(DWTfusion_4,self).__init__()

#         self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
#         self.iwt_img = DWTInverse(wave='haar', mode='zero')
#         self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
#         self.LayerNorm_event=LayerNorm_channel(channel=channel)   
#         self.LayerNorm_image=LayerNorm_channel(channel=channel)   

#         self.conv_img_LL = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(channel,channel,kernel_size=1)
#         )


#         self.H = EventImage_DWT_Fusion(channel*3,num_heads=num_heads)
        
        
#     def forward(self,img_,event_):

#         event = self.LayerNorm_event(event_)
#         img = self.LayerNorm_image(img_)


#         img_LL,img_H = self.dwt_img(img) #img_H包括其他分量
#         img_H = img_H[0]
#         img_H = rearrange(img_H, 'b c fp h w -> b (c fp) h w')

#         event_LL,event_H = self.dwt_event(event) #img_H包括其他分量
#         event_H = event_H[0]
#         event_H = rearrange(event_H, 'b c fp h w -> b (c fp) h w')
        

#         img_LL = self.conv_img_LL(img_LL)

#         fusion_fea_H = self.H(img_H,event_H)
#         fusion_fea_H = rearrange(fusion_fea_H, 'b (c fp) h w -> b c fp h w',fp=3)
        
#         Whlist = [fusion_fea_H]

#         y = self.iwt_img((img_LL, Whlist))

#         return y





class DWTfusion_5(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_5,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        self.conv_img_LL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )


        self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)


        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #img_H包括其他分量

        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        img_LL = self.conv_img_LL(img_LL)

        fusion_fea_LH = self.LH(img_LH,event_LH)
        fusion_fea_HL = self.HL(img_HL,event_HL)
        fusion_fea_HH = self.HH(img_HH,event_HH)


        Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)

        Whlist = [Wh]

        y = self.iwt_img((img_LL, Whlist))

        return y



class DWTfusion_6(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_6,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        self.conv_img_LL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_LH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )


        self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.LH_l = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL_l = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH_l = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)


        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #img_H包括其他分量

        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        #低频调节
        img_LL = self.conv_img_LL(img_LL)
        #高频调节
        #同频率分量调节
        fusion_fea_LH = self.LH(img_LH,event_LH)
        fusion_fea_HL = self.HL(img_HL,event_HL)
        fusion_fea_HH = self.HH(img_HH,event_HH)
        #跨频率分量调节
        l_LH = self.conv_LH(event_LL)
        l_HL = self.conv_HL(event_LL)
        l_HH = self.conv_HH(event_LL)

        fusion_fea_LH = self.LH_l(fusion_fea_LH,l_LH)
        fusion_fea_HL = self.HL_l(fusion_fea_HL,l_HL)
        fusion_fea_HH = self.HH_l(fusion_fea_HH,l_HH)

        Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)

        Whlist = [Wh]

        y = self.iwt_img((img_LL, Whlist))

        return y

#并联同通道和跨通道融合
class DWTfusion_cross(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_cross,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*3,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_LH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )


        self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.LL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)


        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #event_H包括其他分量

        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        #同频率分量调节
        #低频调节
        fusion_fea_LL = self.LL(img_LL,event_LL)
        #高频调节  
        fusion_fea_LH = self.LH(img_LH,event_LH)
        fusion_fea_HL = self.HL(img_HL,event_HL)
        fusion_fea_HH = self.HH(img_HH,event_HH)

        #跨频率分量调节(Cross frequency sub-band fusion中没有输入事件)
        l_LL = self.conv_LL(torch.cat((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=1))
        l_LH = self.conv_LH(fusion_fea_LL)
        l_HL = self.conv_HL(fusion_fea_LL)
        l_HH = self.conv_HH(fusion_fea_LL)

        fusion_fea_LL = self.LL_cross(fusion_fea_LL,l_LL)
        fusion_fea_LH = self.LH_cross(fusion_fea_LH,l_LH)
        fusion_fea_HL = self.HL_cross(fusion_fea_HL,l_HL)
        fusion_fea_HH = self.HH_cross(fusion_fea_HH,l_HH)

        Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)

        Whlist = [Wh]

        y = self.iwt_img((fusion_fea_LL, Whlist))

        return y

#跨通道部分加入事件引导
class DWTfusion_cross_1(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_cross_1,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*4,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_LH = nn.Sequential(
            nn.Conv2d(channel*4,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HL = nn.Sequential(
            nn.Conv2d(channel*4,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(channel*4,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )


        self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.LL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)


        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #event_H包括其他分量

        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        #同频率分量调节
        #低频调节
        fusion_fea_LL = self.LL(img_LL,event_LL)
        #高频调节  
        fusion_fea_LH = self.LH(img_LH,event_LH)
        fusion_fea_HL = self.HL(img_HL,event_HL)
        fusion_fea_HH = self.HH(img_HH,event_HH)

        #跨频率分量调节(Cross frequency sub-band fusion中有输入事件)
        l_LL = self.conv_LL(torch.cat((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH,event_LL),dim=1))
        l_LH = self.conv_LH(torch.cat((fusion_fea_LL,event_LH,event_HL,event_HH),dim=1))
        l_HL = self.conv_HL(torch.cat((fusion_fea_LL,event_LH,event_HL,event_HH),dim=1))
        l_HH = self.conv_HH(torch.cat((fusion_fea_LL,event_LH,event_HL,event_HH),dim=1))

        fusion_fea_LL = self.LL_cross(fusion_fea_LL,l_LL)
        fusion_fea_LH = self.LH_cross(fusion_fea_LH,l_LH)
        fusion_fea_HL = self.HL_cross(fusion_fea_HL,l_HL)
        fusion_fea_HH = self.HH_cross(fusion_fea_HH,l_HH)

        Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)

        Whlist = [Wh]

        y = self.iwt_img((fusion_fea_LL, Whlist))

        return y


class DWTfusion_cross_2(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_cross_2,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event_H=LayerNorm_channel(channel=channel)   
        self.LayerNorm_event_L=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        self.conv_LL = nn.Sequential(
            nn.Conv2d(channel*3,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_LH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )
        self.conv_HH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )


        self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.LL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.LH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HL_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.HH_cross = EventImage_DWT_Fusion(channel,num_heads=num_heads)

        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)
        
    def forward(self,img_,event_l,event_h):

        event_l_ = self.LayerNorm_event_L(event_l)
        event_h_ = self.LayerNorm_event_L(event_h)
        img = self.LayerNorm_image(img_)

        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量
        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        _,event_H = self.dwt_event(event_h_) 
        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        event_LL,_ = self.dwt_event(event_l_) 
        # event_H_ = event_H[0]
        # event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        # event_LH = event_LH.squeeze(dim = 2)
        # event_HL = event_HL.squeeze(dim = 2)
        # event_HH = event_HH.squeeze(dim = 2)

        #同频率分量调节
        #低频调节
        fusion_fea_LL = self.LL(img_LL,event_LL)
        #高频调节  
        fusion_fea_LH = self.LH(img_LH,event_LH)
        fusion_fea_HL = self.HL(img_HL,event_HL)
        fusion_fea_HH = self.HH(img_HH,event_HH)

        #跨频率分量调节(Cross frequency sub-band fusion中没有输入事件)
        l_LL = self.conv_LL(torch.cat((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=1))
        l_LH = self.conv_LH(fusion_fea_LL)
        l_HL = self.conv_HL(fusion_fea_LL)
        l_HH = self.conv_HH(fusion_fea_LL)

        fusion_fea_LL = self.LL_cross(fusion_fea_LL,l_LL)
        fusion_fea_LH = self.LH_cross(fusion_fea_LH,l_LH)
        fusion_fea_HL = self.HL_cross(fusion_fea_HL,l_HL)
        fusion_fea_HH = self.HH_cross(fusion_fea_HH,l_HH)

        Wh = torch.stack((fusion_fea_LH,fusion_fea_HL,fusion_fea_HH),dim=2) #B C 3 H W
        b,c,fp,h,w = Wh.shape
        Wh = rearrange(Wh, 'b c fp h w -> b (fp h w) c')
        Wh = Wh + self.ffn(self.norm2(Wh))
        Wh = rearrange(Wh, 'b (fp h w) c -> b c fp h w',fp = fp,h=h,w=w)

        Whlist = [Wh]

        y = self.iwt_img((fusion_fea_LL, Whlist))

        return y


class DWTfusion_LL(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_LL,self).__init__()
        
        # print('flag')

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        # self.LL = Mutual_Attention(channel,num_heads=num_heads)
        self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        
        
        self.conv_img_HL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_img_LH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_img_HH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)

        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #img_H包括其他分量

        img_LH = self.conv_img_LH(img_LH)
        img_HL = self.conv_img_HL(img_HL)
        img_HH = self.conv_img_HH(img_HH)

        fusion_fea_LL = self.LL(img_LL,event_LL)

        Wh = torch.stack((img_LH,img_HL,img_HH),dim=2)
        Whlist = [Wh]

        y = self.iwt_img((fusion_fea_LL, Whlist))

        return y

class DWTfusion_LL_1(nn.Module):
    def __init__(self,channel,num_heads):
        super(DWTfusion_LL_1,self).__init__()
        
        # print('flag')

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero')
        self.dwt_event = DWTForward(J=1, wave='haar', mode='zero')
        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        # self.LL = Mutual_Attention(channel,num_heads=num_heads)
        self.LL = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        self.cross_H = EventImage_DWT_Fusion(channel,num_heads=num_heads)
        
        
        self.conv_img_HL = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_img_LH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_img_HH = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        self.conv_H = nn.Sequential(
            nn.Conv2d(channel*3,channel,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel,channel,kernel_size=1)
        )

        
        
    def forward(self,img_,event_):

        event = self.LayerNorm_event(event_)
        img = self.LayerNorm_image(img_)

        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)

        event_LL,event_H = self.dwt_event(event) #event_H包括其他分量

        event_H_ = event_H[0]
        event_LH,event_HL,event_HH = torch.chunk(event_H_,3,dim=2)
        event_LH = event_LH.squeeze(dim = 2)
        event_HL = event_HL.squeeze(dim = 2)
        event_HH = event_HH.squeeze(dim = 2)

        #高频处理
        img_LH_c = self.conv_img_LH(img_LH)
        img_HL_c = self.conv_img_HL(img_HL)
        img_HH_c = self.conv_img_HH(img_HH)
        #低频处理
        #同频率分量处理
        fusion_fea_LL = self.LL(img_LL,event_LL)
        #跨频率分量处理
        f_H = self.conv_H(torch.cat((event_LH,event_HL,event_HH),dim=1))
        fusion_fea_LL = self.cross_H(fusion_fea_LL,f_H)

        Wh = torch.stack((img_LH_c,img_HL_c,img_HH_c),dim=2)
        Whlist = [Wh]

        y = self.iwt_img((fusion_fea_LL, Whlist))

        return y


# class CrossChannel_EventSelect(nn.Module):
#     def __init__(self,channel):
#         super(CrossChannel_EventSelect,self).__init__()

#         self.LayerNorm_event=LayerNorm_channel(channel=channel)   
#         self.LayerNorm_image=LayerNorm_channel(channel=channel)
#         self.img_mult = nn.Sequential(
#             nn.Conv2d(channel,channel*2,kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channel*2,channel*2,kernel_size=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channel*2,channel,kernel_size=1),
#             nn.Sigmoid()
#         )

#         self.event_adjust = nn.Sequential(
#             nn.Conv2d(channel,channel,kernel_size=1,groups=channel),
#             nn.ReLU(),
#         )


#         self.norm2 = nn.LayerNorm(channel)
#         self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)




#     def forward(self,img,event):

#         event = self.LayerNorm_event(event)
#         img = self.LayerNorm_image(img)

#         img_c = self.img_mult(img)
#         event_c = self.event_adjust(img_c * event) + event

#         b,c,h,w = event_c.shape
#         event_c = to_3d(event_c)
#         event_c = event_c + self.ffn(self.norm2(event_c))
#         event_c = to_4d(event_c, h, w)

#         return event_c


class CrossChannel_EventSelect(nn.Module):
    def __init__(self,channel):
        super(CrossChannel_EventSelect,self).__init__()

        self.LayerNorm_event=LayerNorm_channel(channel=channel)   
        self.LayerNorm_image=LayerNorm_channel(channel=channel)
        self.mult = nn.Sequential(
            nn.Conv2d(channel*2,channel*2,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*2,channel*2,kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel*2,channel,kernel_size=1),
            nn.Sigmoid()
        )

        self.event_adjust = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        )


        self.norm2 = nn.LayerNorm(channel)
        self.ffn = Mlp(in_features=channel, hidden_features=channel*2, act_layer=nn.GELU, drop=0.)




    def forward(self,img,event):

        event = self.LayerNorm_event(event)
        img = self.LayerNorm_image(img)
        x = torch.cat((img,event),dim=1)

        x = self.mult(x)
        event_c = self.event_adjust(x * event) + event

        b,c,h,w = event_c.shape
        event_c = to_3d(event_c)
        event_c = event_c + self.ffn(self.norm2(event_c))
        event_c = to_4d(event_c, h, w)

        return event_c



#采用自注意力在小波变换域
class DWT_attenion_conv(nn.Module):
    def __init__(self,channel,hidden_channel_factor=2):
        super(DWT_attenion_conv,self).__init__()

        hidden_channel=int(channel*hidden_channel_factor)

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero') 

        self.conv_f1=nn.Conv2d(channel,hidden_channel*2,kernel_size=1)
        self.conv_f2=nn.Conv2d(hidden_channel*2,hidden_channel*2,kernel_size=1)

        self.conv_H = nn.Sequential(
            nn.Conv2d(hidden_channel*6, hidden_channel*6, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*6, hidden_channel*6, kernel_size=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(hidden_channel*6, hidden_channel*3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*3, hidden_channel*6, 1, bias=False),
            nn.Softmax(dim=1)
            )

        self.conv_L = nn.Sequential(
            nn.Conv2d(hidden_channel*2, hidden_channel*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel*2, hidden_channel*2, kernel_size=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(hidden_channel*2, hidden_channel, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel*2, 1, bias=False),
            nn.Softmax(dim=1)
            )
        
        self.out=nn.Conv2d(hidden_channel,channel,kernel_size=1)


    def forward(self,x):


        #频域
        x_f=self.conv_f1(x)
        
        b,c,h,w=x_f.shape
        x_LL,x_H =self.dwt_img(x_f)

        x_H = x_H[0]
        x_H = rearrange(x_H, 'b c fp h w -> b (c fp) h w')
        x_H_atten = self.conv_H(x_H)
        x_H = x_H_atten * x_H
        x_H = rearrange(x_H, 'b (c fp) h w -> b c fp h w',fp=3)

        x_LL_atten = self.conv_L(x_LL)
        x_LL = x_LL_atten*x_LL

        Whlist = [x_H]

        x_f = self.iwt_img((x_LL, Whlist))
        
        
        x_f1,x_f2=self.conv_f2(x_f).chunk(2, dim=1)
        x_f = f.gelu(x_f1) * x_f2
        x_f = self.out(x_f)


        return x_f


class DWTenhanceblock_selfatten(nn.Module):
    def __init__(self,channel,num_heads=1):
        super(DWTenhanceblock_selfatten,self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_k = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(channel, channel, kernel_size=1)

        self.project_out = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):

        b,c,h,w = x.shape

        q = self.conv_q(x) # image
        k = self.conv_k(x) # event
        v = self.conv_v(x) # event   #change!!!

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   #change

        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class DWTenhanceblock_conv(nn.Module):
    def __init__(self,channel):
        super(DWTenhanceblock_conv,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU()
            )
        self.mult = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
            )
        self.sum = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU()
            )
        

    def forward(self, input):
        x = self.conv(input)
        x_mult = self.mult(x)
        x_sum = self.sum(x)

        out = x_mult*x+x_sum + input

        return out


class DWTenhance(nn.Module):
    def __init__(self,channel,heads=1):#
        super(DWTenhance,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero') 
        self.LayerNorm_image=LayerNorm_channel(channel=channel)   

        # self.conv_img_LL = nn.Sequential(
        #     nn.Conv2d(channel,channel,kernel_size=1),
        #     nn.InstanceNorm2d(channel),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(channel,channel,kernel_size=1)
        # )

        self.LL = DWTenhanceblock_selfatten(channel,heads)
        self.H = DWTenhanceblock_selfatten(channel,heads)
        # self.LH = DWTenhanceblock_selfatten(channel,heads)
        # self.HL = DWTenhanceblock_selfatten(channel,heads)
        # self.HH = DWTenhanceblock_selfatten(channel,heads)

        # self.LL = DWTenhanceblock_conv(channel)
        # self.LH = DWTenhanceblock_conv(channel)
        # self.HL = DWTenhanceblock_conv(channel)
        # self.HH = DWTenhanceblock_conv(channel)
        
        
    def forward(self,img):


        # img = self.LayerNorm_image(img_)

        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        img_LH = img_LH.squeeze(dim = 2)
        img_HL = img_HL.squeeze(dim = 2)
        img_HH = img_HH.squeeze(dim = 2)


        img_LL = self.LL(img_LL)
        img_LH = self.H(img_LH)
        img_HL = self.H(img_HL)
        img_HH = self.H(img_HH)


        Wh = torch.stack((img_LH,img_HL,img_HH),dim=2)
        Whlist = [Wh]

        y = self.iwt_img((img_LL, Whlist))

        return y

class DWTenhance_0(nn.Module):
    def __init__(self,channel,heads=1):#
        super(DWTenhance_0,self).__init__()

        self.dwt_img = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt_img = DWTInverse(wave='haar', mode='zero') 

        self.LL = DWTenhanceblock_selfatten(channel,heads)
        self.H = DWTenhanceblock_selfatten(channel*3,heads)

        
        
    def forward(self,img):


        img_LL,img_H = self.dwt_img(img) #img_H包括其他分量

        img_H_ = img_H[0]
        img_H_ = rearrange(img_H_, 'b c fp h w -> b (c fp) h w')
        # img_LH,img_HL,img_HH = torch.chunk(img_H_,3,dim=2)
        # img_LH = img_LH.squeeze(dim = 2)
        # img_HL = img_HL.squeeze(dim = 2)
        # img_HH = img_HH.squeeze(dim = 2)
        # img_h = torch.cat((img_LH,img_HL,img_HH),dim=1)


        img_LL = self.LL(img_LL)
        img_H_ = self.H(img_H_)
        img_H_ = rearrange(img_H_, 'b (c fp) h w -> b c fp h w',fp=3)

        # img_LH,img_HL,img_HH = torch.chunk(img_h,3,dim=1)
        # Wh = torch.stack((img_LH,img_HL,img_HH),dim=2)
        Whlist = [img_H_]

        y = self.iwt_img((img_LL, Whlist))

        return y


class Level1Fusion(nn.Module):
    def __init__(self,c_lv0,c_lv1,c_lv2):
        super(Level1Fusion,self).__init__()

        channel = c_lv1

        self.level0_conv = nn.Sequential(
            nn.Conv2d(c_lv0, channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU()
            )

        self.level1_conv = nn.Sequential(
            nn.Conv2d(c_lv1, channel, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU()
            )

        
        self.level2_conv = nn.Sequential(
            UpsampleLayer(c_lv2,channel,kernel_size=3,padding=1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.ReLU()
            )

        self.fusion = nn.Sequential(
            nn.Conv2d(channel*3, channel*3, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*3, channel, kernel_size=1),
            nn.ReLU()
            )

    def forward(self,lv0,lv1,lv2):
        lv0_=self.level0_conv(lv0)
        lv1_=self.level1_conv(lv1)
        lv2_=self.level2_conv(lv2)

        x = torch.cat((lv0_,lv1_,lv2_),dim=1)

        x = self.fusion(x)

        return x


# class Level0Fusion(nn.Module):
#     def __init__(self,c_lv0,c_lv1,c_lv2):
#         super(Level0Fusion,self).__init__()
        
#         channel = c_lv0
#         self.level0_conv = nn.Sequential(
#             nn.Conv2d(c_lv0, channel, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channel, channel, kernel_size=1),
#             nn.ReLU()
#             )

#         self.level1_conv = nn.Sequential(
#             UpsampleLayer(c_lv1,channel,kernel_size=3,padding=1),
#             nn.Conv2d(channel, channel, kernel_size=1),
#             nn.ReLU()
#             )

        
#         self.level2_conv = nn.Sequential(
#             UpsampleLayer(c_lv2,channel,kernel_size=3,padding=1),
#             UpsampleLayer(channel,channel,kernel_size=3,padding=1),
#             nn.Conv2d(channel, channel, kernel_size=1),
#             nn.ReLU()
#             )

#         self.fusion = nn.Sequential(
#             nn.Conv2d(channel*3, channel*3, kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channel*3, channel, kernel_size=1),
#             nn.ReLU()
#             )

#     def forward(self,lv0,lv1,lv2):
#         lv0_=self.level0_conv(lv0)
#         lv1_=self.level1_conv(lv1)
#         lv2_=self.level2_conv(lv2)

#         x = torch.cat((lv0_,lv1_,lv2_),dim=1)

#         x = self.fusion(x)

#         return x

class MFB(nn.Module):
    #利用encoder输出的多尺度信息进行融合
    def __init__(self,channel=(32,64,128)):
        super(MFB,self).__init__()

        self.sfusion0=EventImage_Fusion_Block(channel=channel[0],num_heads=4)
        self.sfusion1=EventImage_Fusion_Block(channel=channel[1],num_heads=8)
        self.sfusion2=EventImage_Fusion_Block(channel=channel[2],num_heads=16)

        self.channelatten0=ChannelAttention_Block(channel[0])
        self.channelatten1=ChannelAttention_Block(channel[1])
        self.channelatten2=ChannelAttention_Block(channel[2])

        self.cnn0 = nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[0], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1)
            )

        self.cnn1 = nn.Sequential(
            nn.Conv2d(channel[1], channel[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[1], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[2], kernel_size=3, stride=2, padding=1)
            )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(channel[2], channel[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[2], channel[2], kernel_size=1, padding=0),
            nn.ReLU()
            )
        
        self.res0 =ResBlock(channel[2])
        self.res1 =ResBlock(channel[2])
        self.res2 =ResBlock(channel[2])


    def forward(self,img_list,event_list):

        fea0=self.sfusion0(img_list[0],self.channelatten0(event_list[0]))
        fea1=self.sfusion1(img_list[1],self.channelatten1(event_list[1]))
        fea2=self.sfusion2(img_list[2],self.channelatten2(event_list[2]))

        out = self.cnn2(fea2 + self.cnn1(fea1 + self.cnn0(fea0)))
        out = self.res0(out)
        out = self.res1(out)
        out = self.res2(out)

        feature_list = [fea0,fea1,fea2]

        return out,feature_list


class MFB_f(nn.Module):
    #利用encoder输出的多尺度信息进行融合
    def __init__(self,channel=(32,64,128)):
        super(MFB_f,self).__init__()

        self.sfusion0=EventImage_Fusion_Block(channel=channel[0],num_heads=4)
        self.sfusion1=EventImage_Fusion_Block(channel=channel[1],num_heads=8)
        self.sfusion2=EventImage_Fusion_Block(channel=channel[2],num_heads=16)

        self.ffusion0=FourierChannelAtention(channel=channel[0],num_head=1)
        self.ffusion1=FourierChannelAtention(channel=channel[1],num_head=2)
        self.ffusion2=FourierChannelAtention(channel=channel[2],num_head=2)

        self.channelatten0=ChannelAttention_Block(channel[0])
        self.channelatten1=ChannelAttention_Block(channel[1])
        self.channelatten2=ChannelAttention_Block(channel[2])

        self.cnn0 = nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[0], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1)
            )

        self.cnn1 = nn.Sequential(
            nn.Conv2d(channel[1], channel[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[1], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[2], kernel_size=3, stride=2, padding=1)
            )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(channel[2], channel[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[2], channel[2], kernel_size=1, padding=0),
            nn.ReLU()
            )

        self.res0 =ResBlock(channel[2])
        self.res1 =ResBlock(channel[2])
        self.res2 =ResBlock(channel[2])
        


    def forward(self,img_list,event_list):

        event_0 = self.channelatten0(event_list[0])
        fea0=self.sfusion0(img_list[0],event_0)
        fea0=self.ffusion0(fea0,event_0)

        event_1 = self.channelatten1(event_list[1])
        fea1=self.sfusion1(img_list[1],event_1)
        fea1=self.ffusion1(fea1,event_1)

        event_2 = self.channelatten2(event_list[2])
        fea2=self.sfusion2(img_list[2],event_2)
        fea2=self.ffusion2(fea2,event_2)

        out = self.cnn2(fea2 + self.cnn1(fea1 + self.cnn0(fea0)))
        out = self.res0(out)
        out = self.res1(out)
        out = self.res2(out)

        feature_list = [fea0,fea1,fea2]

        return out,feature_list


class MFB_h(nn.Module):
    #利用encoder输出的多尺度信息进行融合
    def __init__(self,channel=(32,64,128)):
        super(MFB_h,self).__init__()

        self.sfusion00=EventImage_Fusion_Block_0(channel=channel[0],num_heads=4)
        self.sfusion01=EventImage_Fusion_Block_0(channel=channel[1],num_heads=8)
        self.sfusion02=EventImage_Fusion_Block_0(channel=channel[2],num_heads=16)
        self.channelatten00=ChannelAttention_Block(channel[0])
        self.channelatten01=ChannelAttention_Block(channel[1])
        self.channelatten02=ChannelAttention_Block(channel[2])


        self.sfusion10=EventImage_Fusion_Block_0(channel=channel[0],num_heads=4)
        self.sfusion11=EventImage_Fusion_Block_0(channel=channel[1],num_heads=8)
        self.sfusion12=EventImage_Fusion_Block_0(channel=channel[2],num_heads=16)
        self.channelatten10=ChannelAttention_Block(channel[0])
        self.channelatten11=ChannelAttention_Block(channel[1])
        self.channelatten12=ChannelAttention_Block(channel[2])

        self.cnn0 = nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[0], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1)
            )

        self.cnn1 = nn.Sequential(
            nn.Conv2d(channel[1], channel[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[1], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channel[1], channel[2], kernel_size=3, stride=2, padding=1)
            )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(channel[2], channel[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel[2], channel[2], kernel_size=1, padding=0),
            nn.ReLU()
            )
        


    def forward(self,img_list,event_list):


        fea0=self.sfusion00(img_list[0],self.channelatten00(event_list[0]))
        fea1=self.sfusion01(img_list[1],self.channelatten01(event_list[1]))
        fea2=self.sfusion02(img_list[2],self.channelatten02(event_list[2]))

        fea0=self.sfusion10(fea0,self.channelatten10(img_list[0]))
        fea1=self.sfusion11(fea1,self.channelatten11(img_list[1]))
        fea2=self.sfusion12(fea2,self.channelatten12(img_list[2]))
        

        out = self.cnn2(fea2 + self.cnn1(fea1 + self.cnn0(fea0)))

        feature_list = [fea0,fea1,fea2]

        return out,feature_list



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



class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(UpsampleLayer, self).__init__()


        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out = self.activation(out)

        return out



class FourierChannelAtention(nn.Module):

    def __init__(self,channel,num_head):
        super(FourierChannelAtention,self).__init__()

        self.num_heads = num_head

        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1))

        self.conv_event = nn.Sequential(
            nn.Conv2d(channel,channel*num_head,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1)
        )

        self.conv_img = nn.Sequential(
            nn.Conv2d(channel,channel*num_head,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1),
            nn.LeakyReLU()
        )

        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))

        self.conv_q = nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1)
        self.conv_k = nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1)
        self.conv_v = nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1)

        self.project_out = nn.Sequential(
            nn.Conv2d(channel*num_head,channel*num_head,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel*num_head,channel,kernel_size=1),
            nn.Softmax(dim=1)
        )

        

    
    def forward(self,img,event):

        img_ = self.conv_img(img)
        event_ = self.conv_event(event)

        img_f=torch.fft.rfft2(img_)
        event_f=torch.fft.rfft2(event_)
        img_amp = torch.abs(img_f)
        event_amp = torch.abs(event_f)

        img_amp = self.conv1(img_amp)
        event_amp = self.conv2(event_amp)
        img_gap = self.gap1(img_amp)
        event_gap = self.gap2(event_amp)

        q = self.conv_q(img_gap) # image
        k = self.conv_k(event_gap) # event
        v = self.conv_v(event_gap) # event   #change!!!

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = f.normalize(q, dim=-2)
        k = f.normalize(k, dim=-2)

        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim=-1)
        ca = (attn @ v)
        ca = rearrange(ca, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=1, w=1)
        ca = self.project_out(ca)

        out = img*ca + img

        return out


class FourierChannelAtention_sum(nn.Module):

    def __init__(self,channel,extend):
        super(FourierChannelAtention_sum,self).__init__()


        self.conv_event = nn.Sequential(
            nn.Conv2d(channel,channel*extend,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*extend,channel*extend,kernel_size=1)
        )

        self.conv_img = nn.Sequential(
            nn.Conv2d(channel,channel*extend,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(channel*extend,channel*extend,kernel_size=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel*extend,channel*extend,kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel*extend,channel*extend,kernel_size=1)
        )

        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))

        self.soft = nn.Softmax(dim=1)


        self.project_out = nn.Sequential(
            nn.Conv2d(channel*extend,channel*extend,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel*extend,channel,kernel_size=1),
            nn.LeakyReLU(dim=1)
        )

        

    
    def forward(self,img,event):

        img_ = self.conv_img(img)
        event_ = self.conv_event(event)

        img_f=torch.fft.rfft2(img_)
        event_f=torch.fft.rfft2(event_)
        img_amp = torch.abs(img_f)
        event_amp = torch.abs(event_f)

        img_amp = self.conv1(img_amp)
        event_amp = self.conv2(event_amp)
        img_gap = self.gap1(img_amp)
        event_gap = self.gap2(event_amp)

        ca = self.soft(img_amp + event_amp)
        out = img_*ca

        out = self.project_out(out) + img

        return out

        
class AdaptiveFilter(nn.Module):
    def __init__(self,in_c=6,fea_c=64):
        super(AdaptiveFilter,self).__init__()

        print('build AdaptiveFilter')
        self.head = nn.Conv2d(in_c, fea_c, kernel_size=3, padding=1)
        self.res0 = nn.Sequential(
            nn.Conv2d(fea_c, fea_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fea_c, fea_c, kernel_size=5, padding=2),
            nn.InstanceNorm2d(fea_c)
            )
        self.res1 = nn.Sequential(
            nn.Conv2d(fea_c, fea_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fea_c, fea_c, kernel_size=5, padding=2),
            nn.InstanceNorm2d(fea_c),
            )
        self.res2 = nn.Sequential(
            nn.Conv2d(fea_c, fea_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fea_c, fea_c, kernel_size=5, padding=2),
            nn.InstanceNorm2d(fea_c),
            )
        self.predict = nn.Conv2d(fea_c, in_c, kernel_size=3, padding=1)

    def forward(self,input):
        x = self.head(input)
        x = self.res0(x)+x
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.predict(x)

        return x



class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = f.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FourierImageEventFusion(nn.Module):
    def __init__(self, channel, ffn_expansion_factor=1,bias=False):

        super(FourierImageEventFusion, self).__init__()

        hidden_features = int(channel * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = channel
        self.image_in = nn.Conv2d(channel, hidden_features * 2, kernel_size=1, bias=bias)
        self.event_in = nn.Conv2d(channel, hidden_features * 2, kernel_size=1, bias=bias)

        # self.sig = nn.Sigmoid()

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, channel, kernel_size=1, bias=bias)

    def forward(self, img,event):
        img_ = self.image_in(img)
        event_ = self.event_in(event)

        img_patch = rearrange(img_, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        img_patch_fft = torch.fft.rfft2(img_patch.float())

        event_patch = rearrange(event_, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        event_patch_fft = torch.fft.rfft2(event_patch.float())
        
        event_patch_fft = event_patch_fft * self.fft###
        x_patch_fft = img_patch_fft*event_patch_fft + event_patch_fft###
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = f.gelu(x1) * x2
        x = self.project_out(x)
        return x


# class FourierImageEventFusion_1(nn.Module):
#     def __init__(self, channel, ffn_expansion_factor=1,bias=False):

#         super(FourierImageEventFusion_1, self).__init__()

#         hidden_features = int(channel * ffn_expansion_factor)

#         self.patch_size = 8

#         self.dim = channel
#         self.image_in = nn.Conv2d(channel, hidden_features * 2, kernel_size=1, bias=bias)
#         self.event_in = nn.Conv2d(channel, hidden_features * 2, kernel_size=1, bias=bias)

#         # self.sig = nn.Sigmoid()

#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)

#         self.fft_mult = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
#         self.fft_plus = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
#         self.project_out = nn.Conv2d(hidden_features, channel, kernel_size=1, bias=bias)

#     def forward(self, img,event):
#         img_ = self.image_in(img)
#         event_ = self.event_in(event)

#         img_patch = rearrange(img_, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#                             patch2=self.patch_size)
#         img_patch_fft = torch.fft.rfft2(img_patch.float())

#         event_patch = rearrange(event_, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
#                             patch2=self.patch_size)
#         event_patch_fft = torch.fft.rfft2(event_patch.float())
        
#         event_patch_fft_mult = event_patch_fft * self.fft_mult###
#         event_patch_fft_plus = event_patch_fft * self.fft_plus###
#         x_patch_fft = img_patch_fft*event_patch_fft_mult + event_patch_fft_plus###
#         x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
#         x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
#                       patch2=self.patch_size)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)

#         x = f.gelu(x1) * x2
#         x = self.project_out(x)
#         return x