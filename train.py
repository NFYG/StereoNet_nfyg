# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:07:31 2023

@author: NFYG
"""

import torch
# import torch.nn as nn
# from torch.optim import RMSprop
# from torch.autograd import Variable
from model import StererNet
import os
import time
import datetime
from dataloader import read_lst_from_txt, MyDataset

import argparse





parser = argparse.ArgumentParser(description="This python file is StereoNet train program")
parser.add_argument('--maxdisp', type=int ,default=191, help='maxium disparity(default 191)')
# parser.add_argument('--datapath', default='/datasets/sceneflow/',help='datapath')
parser.add_argument('--epochs', type=int, default=40,help='number of epochs to train')
parser.add_argument('--loadmodel', type=int,default=1, help='do we load saved check_points?') # 1为载入存档，0为从头训练
# parser.add_argument('--savemodel', default='/check_points', help='save model path')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='SEED_NUM', help='random seed (default: 1)')
args = parser.parse_args()






# 只有在命令行没有禁用cuda，且检测到cuda硬件存在时，才会使用cuda
args.cuda = (not args.no_cuda) and torch.cuda.is_available()



# 设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同
torch.manual_seed(args.seed)



# 用dataloader根据args.datapath载入训练集数据、测试集数据
train_img_left_lst = read_lst_from_txt("train_img_left_lst.txt")
train_img_right_lst = read_lst_from_txt("train_img_right_lst.txt")
train_disp_left_lst = read_lst_from_txt("train_disp_left_lst.txt")
test_img_left_lst = read_lst_from_txt("test_img_left_lst.txt")
test_img_right_lst = read_lst_from_txt("test_img_right_lst.txt")
test_disp_left_lst = read_lst_from_txt("test_disp_left_lst.txt")

# print(len(train_img_left_lst))
# print(len(train_img_right_lst))
# print(len(train_disp_left_lst))
# print(len(test_img_left_lst))
# print(len(test_img_right_lst))
# print(len(test_disp_left_lst))

MyTrainDataset = MyDataset(train_img_left_lst, train_img_right_lst, train_disp_left_lst)
MyTestDataset = MyDataset(test_img_left_lst, test_img_right_lst, test_disp_left_lst)

batch_size = 2 # 超过2显存就不够了
TrainDataloader = torch.utils.data.DataLoader(MyTrainDataset,batch_size=batch_size,
                                              shuffle=True,num_workers=4,drop_last=False,pin_memory=True)
TestDataloader = torch.utils.data.DataLoader(MyTestDataset,batch_size=batch_size,
                                              shuffle=True,num_workers=4,drop_last=False,pin_memory=True)



# 创建模型，并放到GPU上
if args.cuda:
    model = StererNet(batch_size=batch_size, height=540,width=960, device="cuda:0", maxdisp=args.maxdisp, K=4)
    model.cuda()
else:
    model = StererNet(batch_size=batch_size, height=540,width=960, device="cpu", maxdisp=args.maxdisp, K=4)
    



# 选择RMSprop优化器，可以消除梯度下降时Loss的抖动
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
epoch_start = 0



# 载入模型训练存档，与设置载入点的学习率
proj_root_path = r'D:\myproject\20230714depth_estimate\ROBOVISION_stereonet\StereoNet_s'
check_points_path = proj_root_path + "/check_points/train_save_sceneflow.tar"

if os.path.exists(check_points_path) and (args.loadmodel != 0):
    # 若存档存在且参数要求载入存档，则载入，否则从头训练
    print("-- checkpoint loaded --")
    saved_dict = torch.load(check_points_path)                    # #载入状态字典
    model.load_state_dict(saved_dict["model_state_dict"])         # 载入存档的模型参数
    optimizer.load_state_dict(saved_dict["optimizer_state_dict"]) # 载入存档的优化器参数
    epoch_start = saved_dict["epoch"] + 1                         # 从存档epoch的下一个epoch开始训练
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1) # epoch_start
else:
    print("-- no checkpoint loaded --")                           # last_epoch默认取-1，表明学习率取初始值
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  























# 训练一个batch
def train(img_L, img_R, disp_L):
    
    model.train()         # model.train()的作用是启用 Batch Normalization 和 Dropout
    
    if args.cuda:         # 部署到GPU上
        imgL, imgR, disp_true = img_L.cuda(), img_R.cuda(), disp_L.cuda()

    mask = disp_true < args.maxdisp # 找出那些视差在允许范围之内的位置
    mask.detach_()        # detach_表示不用计算该tensor相对参数的梯度

    output1,output2 = model(imgL, imgR)         # [batch_size,1,H,W]
    
    output1 = torch.squeeze(output1, 1)         # [batch_size,H,W]
    output2 = torch.squeeze(output2, 1)
    # print(output1.requires_grad, output2.requires_grad)
    
    # 参考：https://blog.csdn.net/m0_46653437/article/details/111587462
    # reduction='sum'，返回的是标量
    # loss1 = torch.nn.functional.smooth_l1_loss(output1[mask], disp_true[mask], reduction="sum")
    # loss2 = torch.nn.functional.smooth_l1_loss(output2[mask], disp_true[mask], reduction="sum")
    loss1 = torch.nn.functional.smooth_l1_loss(output1[mask], disp_true[mask], reduction="mean")
    loss2 = torch.nn.functional.smooth_l1_loss(output2[mask], disp_true[mask], reduction="mean")
    loss = loss1 + loss2 
    
    optimizer.zero_grad()              # 优化器梯度清零
    loss.backward(retain_graph=False)  # 反向传播求得梯度
    optimizer.step()                   # 优化器步进更新参数

    return loss.detach().item()        # item()将tensor变成一个标量返回




# 测试一个batch
def test(img_L, img_R, disp_true):
    
    model.eval()          # model.eval()的作用是不启用 Batch Normalization 和 Dropout。
    # imgL   = Variable(torch.FloatTensor(imgL))
    # imgR   = Variable(torch.FloatTensor(imgR))
    
    if args.cuda:
        imgL, imgR = img_L.cuda(), img_R.cuda()

    mask = disp_true < args.maxdisp

    with torch.no_grad():  # 在该with模块下，所有计算得出的tensor的requires_grad都自动设置为False
        output1, output2 = model(imgL,imgR)         # [batch_size,1,H,W]
        output1 = torch.squeeze(output1, 1)         # [batch_size,H,W]
        output2 = torch.squeeze(output2, 1)
        
        output1 = output1.cpu()
        output2 = output2.cpu()
        
    loss1 = torch.nn.functional.smooth_l1_loss(output1[mask], disp_true[mask])
    loss2 = torch.nn.functional.smooth_l1_loss(output2[mask], disp_true[mask])
    loss = loss1 + loss2
    
    EPE =  torch.mean(torch.abs(output2[mask]-disp_true[mask]))  # end-point-error
    
    return loss.detach().item(), EPE






def main_train():
    
    train_start_time = time.time()
    
    # 创建记录训练信息的日志文件
    temp = str(datetime.datetime.now()).split(" ")
    # name = "\log\\" + temp[0] + "_" +temp[1][0:8].replace(":", "_") + "_train_Log.txt"
    name = proj_root_path + "/log/" + temp[0] + "_" +temp[1][0:8].replace(":", "_") + "_train_Log.txt"
    log_file = open(name,'w')
    
    for epoch in range(epoch_start, args.epochs):
        
        log_info = "This is " + str(epoch) + " epoch, lr is " + str(scheduler.get_last_lr()[0])
        log_file.write(log_info+"\n")
        print(log_info)
        
        this_epoch_total_loss = 0              # 用于记录该epoch每次训练的loss的总和
        
        
        
        for i, (img_L, img_R, disp_L) in enumerate(TrainDataloader):
            
            start_time = time.time()           # 记录开始时刻
            loss = train(img_L, img_R, disp_L) # 执行一次训练
            this_epoch_total_loss += loss      # 累加总loss
            
            if(i%20 == 0):                     # 每训练10次打印一次信息
                log_info2 = "epoch:" + str(epoch) + ",batch_idx:" + str(i)[0:7] + ",loss:" + str(loss) \
                          + ",time_consumed(s):" + str(time.time() - start_time)[:6] \
                          + ",learning rate:" + str(optimizer.state_dict()['param_groups'][0]['lr'])
                # for param_group in optimizer.param_groups:
                #     print(param_group['lr'])
                log_file.write(log_info2+"\n")
                print(log_info2)
                
                # 打印参数，验证参数是否得到更新
                # print(model.state_dict()["downsampling.0.bias"])
        
        
        average_loss = this_epoch_total_loss/(len(TrainDataloader))
        log_info3 = "epoch " + str(epoch) + " average train loss is " + str(average_loss)
        log_file.write(log_info3+"\n")
        print(log_info3)                       # 打印一个epoch完毕后的平均loss
        
        scheduler.step()                       # 每当一个epoch训练完毕后，更新学习率
        
        # 每个epoch保存一次，文件名带上日期时间和epoch信息，否则新保存的会覆盖之前保存的
        data_need_be_saved = {"model_state_dict":model.state_dict(),
                              "optimizer_state_dict":optimizer.state_dict(),
                              "epoch":epoch}
        torch.save(data_need_be_saved,check_points_path)
    
    # 所有epoch训练完毕打印总耗时
    log_info4 = "all epoch consume " + str((time.time()-train_start_time)/3600.0)+" hours"
    log_file.write(log_info4+"\n")
    print(log_info4)
    
    # 测试集上进行测试
    total_test_loss, total_EPE = 0, 0
    for i, (img_L, img_R, disp_L) in enumerate(TestDataloader):
        loss,EPE = test(img_L, img_R, disp_L)
        total_test_loss += loss
        total_EPE += EPE
    average_test_loss = total_test_loss/len(TestDataloader)
    average_test_EPE = total_EPE/len(TestDataloader)
    log_info5 = "average_test_loss " + str(average_test_loss) + " average_test_EPE " +str(average_test_EPE)
    log_file.write(log_info5+"\n")
    print(log_info5)
    
    # 最后后关闭日志文件
    log_file.close()
    


























if __name__ == "__main__":
    # print(args.maxdisp)
    # print(args.no_cuda)
    # print(args.seed)
    main_train()
    
    input("END")
    
    
    
    
    
























































