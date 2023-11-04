# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:49:11 2023

@author: NFYG
"""
import torch
import torch.nn as nn
import numpy as np
import time



class StererNet(nn.Module):
    
    def __init__(self, batch_size, height, width, device, maxdisp, K=4):
        
        super().__init__()
        # 创建时需指定batch_size，原文设置为 1，K表示下采样的层数，原文取3或4
        # maxdisp表示最大视差，例如maxdisp=127，则视差范围就是0~127共128个数
        # forward输入是两个tensor，对应左&右2张图片，形状都是[batch_size, 3, height, width]
        # 图片的高&宽相同，默认是RGB图，所以是3通道
        self.batch_size = batch_size
        self.height = height
        self.width = width
        
        self.maxdisp = maxdisp
        self.K = K
        self.device = device
        
        # 部署到GPU上
        if torch.cuda.is_available():
            # print("GPU NUM:", torch.cuda.device_count())
            pass
        else:
            # print("NO GPU available")
            pass
        
        # STEP 1 下采样，用以获取低分辨率的图片特征，是feature extraction的part1
        # 根据论文，中间模块的特征通道数均为32，卷积核为5*5
        # stride取2确保每经过一层conv2d图片的高、宽均减半
        # 降采样可以让下一层卷积有更大的感受野(receptive field)
        # 经过下采样层后输出的tensor形状为[batch_size, 32, height//2^K, width//2^K]
        if(self.K == 4):
            self.downsampling = nn.Sequential(
                nn.Conv2d( 3, 32, 5, stride=2, padding=2),
                nn.Conv2d(32, 32, 5, stride=2, padding=2),
                nn.Conv2d(32, 32, 5, stride=2, padding=2),
                nn.Conv2d(32, 32, 5, stride=2, padding=2)
                )
        elif(self.K == 3):
            self.downsampling = nn.Sequential(
                nn.Conv2d( 3, 32, 5, stride=2, padding=2),
                nn.Conv2d(32, 32, 5, stride=2, padding=2),
                nn.Conv2d(32, 32, 5, stride=2, padding=2)
                )
        else:
            raise ValueError("ERROR : K should be 3 or 4.")
            
        
        
        # STEP 2 原文使用6个残差模块+1个2d卷积。残差模块内部由conv2d+batchnorm2d+leakyrelu构成
        # 是feature extraction的part2，经过残差层后输出的tensor形状为[batch_size, 32, height//2^K, width//2^K]
        self.six_residual_blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1)
            )
        
        
        
        # STEP 3 构建cost volume，论文中提到可用左右feature map相减或者拼接的方式，此处选择相减
        # 在forward中直接调用build_cost_volume即可
        tempH, tempW = self.height, self.width
        for i in range(K):
            # 公式参考：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            tempH = torch.tensor((tempH + 2*2 - 1*(5-1) - 1)/2.0 + 1).floor().int().item()
            tempW = torch.tensor((tempW + 2*2 - 1*(5-1) - 1)/2.0 + 1).floor().int().item()
            
            # if(tempH%2 == 0):
            #     tempH = tempH//2
            # else:
            #     tempH = tempH//2 + 1
                
            # if(tempW%2 == 0):
            #     tempW = tempW//2
            # else:
            #     tempW = tempW//2 + 1
            
        # feature_shape = (self.batch_size, 32, self.height//(2**self.K), self.width//(2**self.K))
        feature_shape = (self.batch_size, 32, tempH, tempW)
        self.build_cost_volume = build_cost_volume(feature_shape, self.maxdisp, self.K, self.device)
        
        
        
        # STEP 4 cost volume filter,4*(conv3d+batchnorm3d+leakyrelu) + conv3d
        # 最后一个单独的Conv3d输出1通道，输出形状为[batchsize, 1, (maxdisp+1)//2^K, H//2^K, W//2^K]
        # 维度为1的维度将被squeeze掉，所以输出形状为[batchsize, (maxdisp+1)//2^K, H//2^K, W//2^K]
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(32,32),
            MetricBlock(32,32),
            MetricBlock(32,32),
            MetricBlock(32,32),
            nn.Conv3d(32, 1, 3, stride=1, padding=1)
            )
        
        
        
        # STEP 5 可微分的arg min，采用softmax
        # 输出形状为[batchsize, 1, H//2^K, W//2^K]
        self.softmax_argmin = Differentiable_argmin((self.maxdisp+1)//(2**self.K), self.device)
        
        
        # STEP 6 edge-preserve refine
        # 输入上采样之后的深度图和RGB左图的拼接，所以是4通道输入，输出的是偏差
        # 要把偏差加到原始深度图上
        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            ResidualBlock(32, 32, dilation=1),
            ResidualBlock(32, 32, dilation=2),
            ResidualBlock(32, 32, dilation=4),
            ResidualBlock(32, 32, dilation=8),
            ResidualBlock(32, 32, dilation=1),
            ResidualBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, 1, 1)
            )
        
        
        
        # STEP 7 第六步refine得到的结果加上第五步argmin插值的结果，最后还要通过一个relu保证视差为正
        self.final_relu = nn.ReLU()
    
    
    
    def forward(self, left, right):
        
        # left :[batchsize, 3, height, width]
        # right:[batchsize, 3, height, width]
        # print("左图形状：",left.shape)
        # print("右图形状：",right.shape)
        
        # 检查输入tensor设备&维度是否正确
        # assert left.device == right.device
        # assert left.device == self.device
        # assert left.shape[2] == self.height,str(self.height)+','+str(left.shape[2])
        # assert left.shape[3] == self.width,str(self.width)+','+str(left.shape[2])
        
        # print(left.device)
        # print(right.device)
        
        # 提取左右图的特征
        # t1 = time.time_ns()
        # feature_L = self.six_residual_blocks(self.downsampling(left))
        # feature_R = self.six_residual_blocks(self.downsampling(right))
        
        # 以下的写法可以减少4ms
        batchsize = left.shape[0] 
        temp = self.six_residual_blocks(self.downsampling(torch.cat([left,right],dim=0)))
        # feature_L = temp[:self.batch_size,:,:,:]
        # feature_R = temp[self.batch_size:,:,:,:]
        feature_L = temp[:batchsize,:,:,:]
        feature_R = temp[batchsize:,:,:,:]
        # print(feature_L.shape)
        # print(feature_R.shape)
        
        # print("特征图形状(左)：",feature_L.shape)
        # print("特征图形状(右)：",feature_R.shape)
        
        # 构建代价体积
        # t2 = time.time_ns()
        # cost_volume = build_cost_volume(feature_L, feature_R, self.maxdisp, self.K, self.device)
        cost_volume = self.build_cost_volume(feature_L, feature_R)
        # print("cost volume:", cost_volume.shape)
        
        # 代价聚合,把特征维度压缩掉
        # t3 = time.time_ns()
        cost_low_resolution = self.cost_volume_filter(cost_volume)
        cost_low_resolution = torch.squeeze(cost_low_resolution,dim=1)
        # print("代价聚合后:", cost_low_resolution.shape)
        
        # 得到低分辨率的最小视差图
        # t4 = time.time_ns()
        disp_low_resolution = self.softmax_argmin(cost_low_resolution)
        # print("低分辨率视差图:", disp_low_resolution.shape)
        
        # 恢复视差的范围，从0~(maxdisp+1)//2^K-1恢复成0~maxdisp
        # t5 = time.time_ns()
        disp_low_resolution = disp_low_resolution*(left.shape[-1]/disp_low_resolution.shape[-1])
        disp_high = nn.functional.interpolate(disp_low_resolution, size=left.shape[-2:],
                                              mode='bilinear', align_corners=False)
        # print("插值后高分辨率:", disp_high.shape)
        # t6 = time.time_ns()
        disp_refine = self.refine(torch.cat((disp_high,left),dim=1))
        # print("disp_refine:", disp_refine.shape)
        
        # 
        # t7 = time.time_ns()
        disp_refined = disp_high + disp_refine
        disp_refined = self.final_relu(disp_refined)
        # print("refined高分辨率:", disp_refined.shape)
        
        # print("1",(t2-t1)*1e-6)
        # print("2",(t3-t2)*1e-6)
        # print("3",(t4-t3)*1e-6)
        # print("4",(t5-t4)*1e-6)
        # print("5",(t6-t5)*1e-6)
        # print("6",(t7-t6)*1e-6)
        
        # ret = torch.cat([disp_high,disp_refined], dim=1)
        # return ret
        return disp_high, disp_refined
        # return disp_high
        






class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, dilation=1, stride=1):
        
        super().__init__()
        # kernel_size固定为3，stride固定为1，padding要根据dilation计算得到
        padding = dilation
        
        self.conv2d_1 = nn.Conv2d( in_channel, out_channel, kernel_size=3,
                                  stride=stride, padding=padding, dilation=dilation)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                                  stride=stride, padding=padding, dilation=dilation)
        
        self.batchnorm2d_1 = nn.BatchNorm2d(out_channel)
        self.batchnorm2d_2 = nn.BatchNorm2d(out_channel)
        
        self.leaky_relu_1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu_2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        
        # x形状为[batch_size, 32, height//2^K, width//2^K]
        
        out = self.conv2d_1(x)
        out = self.batchnorm2d_1(out)
        out = self.leaky_relu_1(out)
        
        out = self.conv2d_2(out)
        out = self.batchnorm2d_2(out)
        out = self.leaky_relu_2(out+x) # 残差网络，最终通过激活函数前要加上输入x
        
        return out





class MetricBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        
        super().__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.batchnorm3d = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(negative_slope=0.2,inplace=True) # 负半轴斜率仍旧是0.2
    
    def forward(self, x):
        
        out = self.conv3d(x)
        out = self.batchnorm3d(out)
        out = self.relu(out)
        
        return out





class Differentiable_argmin(nn.Module):
    
    def __init__(self, disp, device):
        '''
        note:不要在__init__中定义计算图中涉及的tensor，要在forward中定义
        '''
        super().__init__()
        
        self.device = device
        self.disp = disp

    
    def forward(self, x):
        
        # disp代表视差的数目
        # self.disp_tensor = torch.FloatTensor(np.reshape(
        #     np.array(range(self.disp)), [1, self.disp, 1, 1])).to(self.device)
        # 20231026,FloatTensor是默认转成FP32的,但我们这里要求dtype跟随输入x的类型，因为输入有可能是FP16的
        self.disp_tensor = torch.tensor(np.reshape(
            np.array(range(self.disp)), [1, self.disp, 1, 1]),dtype=x.dtype).to(self.device)
        
        # x的形状为[batchsize, (maxdisp+1)//2^K, H//2^K, W//2^K]
        # assert x.shape[1] == self.disp_tensor.shape[1]
        
        disp_tensor = self.disp_tensor.repeat((x.shape[0], 1, x.shape[2], x.shape[3]))
        out = nn.functional.softmax(-x, dim=1) # x代表了各个像素点处各个视差下的匹配代价，代价越大，权重越小，所以要有负号
        out = torch.sum(out*disp_tensor, dim=1, keepdim=True)
        
        return out



    

class build_cost_volume(nn.Module):
    
    def __init__(self, feature_shape, maxdisp, K, device):
        '''
        note:不要在__init__中定义计算图中涉及的tensor，要在forward中定义
        '''
        super().__init__()
        # feature_map_L和feature_map_R的形状为[batch_size, 32, height//2^K, width//2^K]
        
        # 下采样后，视差的范围也要相应地减少
        self.disp_low_resolution = (maxdisp+1)//(2**K)
        self.feature_shape = feature_shape
        self.device = device
        

    
    def forward(self, feature_map_L, feature_map_R):
        
        
        feature_shape = self.feature_shape
        
        # 输出的cost_volume的形状是[batch_size, 32, (maxdisp+1)//(2^K), height//2^K, width//2^K]
        self.cost_volume = torch.zeros((feature_shape[0], 
                                       feature_shape[1], 
                                       self.disp_low_resolution,
                                       feature_shape[2],
                                       feature_shape[3]), dtype=feature_map_L.dtype).to(self.device)
        # 20231026确保dtype与输入tensor的dtype一致，因为输入的有可能是FP32或FP16
        
        
        for i in range(self.disp_low_resolution):
            if(i == 0):
                # 视差为0的特殊情况
                # print(feature_map_L.shape,feature_map_R.shape)
                self.cost_volume[:,:,i,:,:] = feature_map_L - feature_map_R
            else:
                # 视差大于0，此时左图需左移i个像素后再与右图相减
                self.cost_volume[:,:,i,:,i:] = feature_map_L[:,:,:,i:] - feature_map_R[:,:,:,:-i]
        
        self.cost_volume = self.cost_volume.contiguous()
        # print("cost_volume device:",cost_volume.device)
        
        return self.cost_volume
        
        
        
        
        

        



if __name__ == "__main__":
    h = 540
    w = 960
    datasize = 30
    # left  = torch.randn((1,3,h,w)).cuda()
    # right = torch.randn((1,3,h,w)).cuda()
    left  = torch.ones((datasize,1,3,h,w)).cuda()
    right = torch.ones((datasize,1,3,h,w)).cuda()
    net = StererNet(batch_size=1, height=h, width=w, device=left.device, maxdisp=191, K=4)
    net.cuda()
    
    
    
    #==========================================================================
    # 参考：https://zhuanlan.zhihu.com/p/94971100 <一文梳理pytorch保存和重载模型参数攻略>
    # 打印模型结构与参数形状
    for k,v in net.state_dict().items():
        print(k," "*(60-len(str(k)))," 参数形状：",v.shape)
    # 打印某部分的参数，可以看到参数是随机初始化的
    # print(net.state_dict()["downsampling.0.bias"])
    
    # 将参数保存
    # torch.save(obj=net.state_dict(),f="model_parameters_save.pth")
    
    # 也能以字典形式保存任何数据
    torch.save(obj={"asdasd":123456789,"params":net.state_dict()},f="model_parameters_save123.pth")
    
    # 载入保存的参数
    saved_param_dict = torch.load("model_parameters_save.pth")
    net.load_state_dict(saved_param_dict)
    
    # 载入自定义的各种奇怪数据
    test_dict = torch.load("model_parameters_save123.pth")
    print(test_dict["asdasd"])
    
    # 载入保存的参数后，再度打印，会发现结果固定不变
    print(net.state_dict()["downsampling.0.bias"])
    #==========================================================================
    
    

    for i in range(30):
        t1 = time.time_ns()
        ret = net(left[i,:,:,:,:], right[i,:,:,:,:])
        print(ret[1].shape)
        torch.cuda.synchronize()
        t2 = time.time_ns()
        print(i," 耗时：",(t2-t1)*1e-6)
        # print(ret[0][0,0,:,:])     # 若载入了保存的参数，每次结果都一样，否则每次结果不同，因为参数会随机初始化
        # time.sleep(0.1)

    input("END")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    