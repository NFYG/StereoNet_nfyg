# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:58:40 2023

@author: NFYG
"""

import torch
from model import StererNet
import os
import time
import datetime
from dataloader import read_lst_from_txt, MyDataset
from torchvision import transforms
from torchstat import stat # 统计模型参数量&计算量，对于不太标准的模型很多算子不支持，暂时没啥用
import numpy as np
from PIL import Image







# stage1-创建模型并载入参数
# 模型forward输入是两个tensor，对应左&右2张图片，形状都是[batch_size, 3, height, width]
batch_size = 1 # 我的垃圾电脑,训练时超过2显存就不够了
height = 540
width =  960
maxdisp = 191 # 视差范围就是0~191共192个数
args_cuda = torch.cuda.is_available()
# 创建模型，并放到GPU上
if args_cuda:
    model = StererNet(batch_size=batch_size, height=height, width=width, device="cuda:0", maxdisp=maxdisp, K=4)
    model.cuda()
else:
    model = StererNet(batch_size=batch_size, height=height, width=width, device="cpu", maxdisp=maxdisp, K=4)


# 载入模型参数
proj_root_path = r'D:\myproject\20230714depth_estimate\ROBOVISION_stereonet\StereoNet_s'
check_points_path = proj_root_path + "/check_points/train_save_sceneflow.tar"
if os.path.exists(check_points_path):
    # print("-- checkpoint loaded --")
    saved_dict = torch.load(check_points_path)                    # #载入状态字典
    model.load_state_dict(saved_dict["model_state_dict"])         # 载入存档的模型参数
else:
    assert(False),"cannot find parameters"
    

model.eval()         # model.eval()的作用是关闭 Batch Normalization 和 Dropout

    


# -----------------------------------------------------------------------------



# stage2-载入测试数据集
train_img_left_lst = read_lst_from_txt("train_img_left_lst.txt")
train_img_right_lst = read_lst_from_txt("train_img_right_lst.txt")
train_disp_left_lst = read_lst_from_txt("train_disp_left_lst.txt")
test_img_left_lst = read_lst_from_txt("test_img_left_lst.txt")
test_img_right_lst = read_lst_from_txt("test_img_right_lst.txt")
test_disp_left_lst = read_lst_from_txt("test_disp_left_lst.txt")

MyTestDataset = MyDataset(test_img_left_lst, test_img_right_lst, test_disp_left_lst)
TestDataloader = torch.utils.data.DataLoader(MyTestDataset,batch_size=batch_size,
                                              shuffle=True,num_workers=1,drop_last=False,pin_memory=True)



# -----------------------------------------------------------------------------


if __name__ == '__main__':
    
    # stage3-推断,检验并绘图
    args_half = 0
    if args_half:
        # 将模型参数从fp32转换成fp16，但torch只有GPU才支持半精度运算试了试发现对于此类要求精度的回归问题，直接量化结果很差
        model.half()
    
    total_time, total_epe = 0.0, 0.0
    for i, (img_L, img_R, disp_L) in enumerate(TestDataloader):
       
        if args_cuda:         # 部署到GPU上
            if args_half:
                # 若将模型输入从FP32转FP16,type之后必须重新部署到cuda
                imgL = img_L.type(torch.HalfTensor).cuda()
                imgR = img_R.type(torch.HalfTensor).cuda()
                disp_true = disp_L.type(torch.HalfTensor).cuda()
            else:
                imgL, imgR, disp_true = img_L.cuda(), img_R.cuda(), disp_L.cuda()
        else:
            assert (not args_half),"half tensor cannot be processed in cpu"
            imgL, imgR, disp_true = img_L.cpu(), img_R.cpu(), disp_L.cpu()
        
        
        
        # 裁剪为指定的高宽
        imgL = imgL[:,:,0:height,0:width] 
        imgR = imgR[:,:,0:height,0:width] 
        disp_true = disp_true[:,0:height,0:width] 

        
        disp_true = disp_true.unsqueeze(1)
        mask = disp_true < maxdisp # 找出那些视差在允许范围之内的位置
        mask.detach_()             # detach_表示不用计算该tensor相对参数的梯度
        
        # ---------------------------------------------------------------------
        
        # torch自带的时间统计工具
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            output1,output_refined = model(imgL, imgR)         # [batch_size,1,H,W]
        
        end.record()
        torch.cuda.synchronize()
        
        err = torch.abs(output_refined[mask]-disp_true[mask])
        EPE =  torch.mean(err)  # end-point-error
        if 0: # 打印误差
            # print("最大视差:",torch.max(disp_true))
            err = err.reshape(-1)
            err01 = err < 0.1
            print("达标像素数:",torch.sum(err01).item(),", 占比:",torch.sum(err01).item()/len(err01))
            err03 = err < 0.3
            print("达标像素数:",torch.sum(err03).item(),", 占比:",torch.sum(err03).item()/len(err03))
            err05 = err < 0.5
            print("达标像素数:",torch.sum(err05).item(),", 占比:",torch.sum(err05).item()/len(err05))
            err07 = err < 0.7
            print("达标像素数:",torch.sum(err07).item(),", 占比:",torch.sum(err07).item()/len(err07))
            err10 = err < 1.0
            print("达标像素数:",torch.sum(err10).item(),", 占比:",torch.sum(err10).item()/len(err10))
        
        print("time(ms):",start.elapsed_time(end)," EPE:",EPE.item())
        if i >= 30:
            total_time += start.elapsed_time(end)
            total_epe  += EPE.item()
        if i == 3030 - 1:
            print("平均时间:",total_time/3000.0, ",平均EPE:",total_epe/3000.0)
            input("end")
        
        
        # ---------------------------------------------------------------------
        
        if 0:
            EPE =  torch.mean(torch.abs(output_refined[mask]-disp_true[mask]))  # end-point-error
            print("视差平均error:",EPE.item())
            
            
            imgL = imgL.squeeze().permute(1,2,0)  # [H,W,3]
            imgR = imgR.squeeze().permute(1,2,0)
            imgL = (imgL*0.5 + 0.5)*255.0
            imgR = (imgR*0.5 + 0.5)*255.0
            imgL = imgL.detach().cpu().numpy()
            imgR = imgR.detach().cpu().numpy()
            imgL = np.asarray( imgL, np.uint8 )
            imgR = np.asarray( imgR, np.uint8 )
            imL = Image.fromarray(imgL)
            imR = Image.fromarray(imgR)
            imL.show()
            imR.show()
            
            output_refined = output_refined.squeeze() # 变成[H,W]
            output_refined = output_refined.detach().cpu().numpy() # detach()不再需要grad,cpu()移到内存
            output_refined = np.asarray( output_refined+0.5, np.uint8 )
            im = Image.fromarray(output_refined)
            im.show()
            input("pause")
        # imL.close()
        # imR.close()
        # im.close()
