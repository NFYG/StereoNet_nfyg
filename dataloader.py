# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:33:11 2023

@author: NFYG
"""

import numpy as np
from PIL import Image
import torch.utils.data
from torchvision import transforms
import os



def read_PFM(file_path):
    
    with open(file_path,'rb') as file:
        
        # 读取第一、第二、第三行，strip()为去除头尾空字符，rstrip()为去除尾部空字符
        # 读取得到bytes类型，用decode('utf-8')转化为字符串类型
        header = file.readline().strip()
        dim_info = file.readline().strip().decode('utf-8')
        scale = float(file.readline().strip())
        
        # 读取第一行header，决定该PFM图存储的是RGB还是灰度图
        RGBimg = None
        if(header == b'PF'):
            RGBimg = True
        elif(header == b'Pf'):
            RGBimg = False
        else:
            raise Exception('PFM file 1st line must be PF or Pf.') 
        
        # 读取第二行，得到两个数字的字符串，分别图片的是width和height
        dim_list = dim_info.split(" ")
        width = int(dim_list[0].strip())
        height = int(dim_list[1].strip())
        
        # 读取第三行得到scale，一个float数，根据scale的正负号判断是小端or大端存储
        if(scale < 0):
            endian = '<'   # 小端存储
            scale = -scale
        else:
            endian = '>'   # 大端存储
        
        shape = (height, width, 3) if RGBimg else (height, width)
        img_data = np.fromfile(file, endian+'f') 
        # 此时file指向第四行，'>f'表示按照大端float32读取，'<f'表示按照小端float32读取

        
    img_data = np.reshape(img_data, shape) # 将数组形状变为RGB(height, width, 3)或(灰度图height, width)
    img_data = np.flipud(img_data)         # .PFM文件从下往上、从左往右储存图片，所以读取后需上下颠倒
      
    # 转成uint8类型后，可以直接存为灰度图查看
    # temp = img_data.astype(np.uint8)
    # im = Image.fromarray(temp, mode='L') # mode='L'表示灰度图，'RGB'表示彩色图
    # im.save("pfmread.png")
    
    return img_data, scale



def disp_loader(path):
    data,scale = read_PFM(path)
    return data



def rgbimg_loader(path):
    # from PIL import Image，Image.open读出来的图像是4通道的，RGBA，所以要转成RGB
    # 载入的是一个PIL.Image对象，可以通过size属性直接获取宽高
    return Image.open(path).convert('RGB')
    


def list_all_img(dataset_path):
    
    all_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    
    # 筛选出名字里带有‘disparity’的文件夹，并合成绝对路径
    disp_folders = [os.path.join(dataset_path, folder) for folder in all_folders if(folder.find('disparity')>-1)]
    imag_folders = [os.path.join(dataset_path, folder) for folder in all_folders if(folder.find('frames_finalpass')>-1)]
    # print(disp_folders)
    # print(imag_folders)
    
    train_img_left_lst = []
    train_img_right_lst = []
    train_disp_left_lst = []
    
    test_img_left_lst = []
    test_img_right_lst = []
    test_disp_left_lst = []

    #==========================================================================
    # FlyingThing3D finalpass
    for disp_dir in disp_folders:
         
         sub_lst1 = [each for each in os.listdir(disp_dir)]
         for sub_dir1 in sub_lst1: # 'TEST' 'TRAIN'
             if sub_dir1 == 'TRAIN':
                 disp_dir_train_1 = disp_dir+'/'+'TRAIN'
                 
                 sub_lst2 = [each for each in os.listdir(disp_dir_train_1)]
                 for sub_dir2 in sub_lst2: # 'A' 'B' 'C'
                     disp_dir_train_2 = disp_dir_train_1 + '/' + sub_dir2
                     
                     sub_lst3 = [each for each in os.listdir(disp_dir_train_2)]
                     for sub_dir3 in sub_lst3: # '0000' '0001' '0002' ~ '0749'
                         disp_dir_train_3 = disp_dir_train_2 + '/' + sub_dir3

                         disp_dir_train_3 = disp_dir_train_3 + '/' + 'left'   # 我们只需要left视差图
                         sub_lst4 =  [each for each in os.listdir(disp_dir_train_3)]
                         for pfm_file in sub_lst4: # '0006.pfm'~'0015.pfm'
                             whole_file_path = disp_dir_train_3 + '/' + pfm_file
                             train_disp_left_lst.append(whole_file_path)
                             #print(whole_file_path)
                 
             elif sub_dir1 == 'TEST':
                 disp_dir_train_1 = disp_dir+'/'+'TEST'
                 
                 sub_lst2 = [each for each in os.listdir(disp_dir_train_1)]
                 for sub_dir2 in sub_lst2: # 'A' 'B' 'C'
                     disp_dir_train_2 = disp_dir_train_1 + '/' + sub_dir2
                     
                     sub_lst3 = [each for each in os.listdir(disp_dir_train_2)]
                     for sub_dir3 in sub_lst3: # '0000' '0001' '0002' ~ '0149'
                         disp_dir_train_3 = disp_dir_train_2 + '/' + sub_dir3

                         disp_dir_train_3 = disp_dir_train_3 + '/' + 'left'   # 我们只需要left视差图
                         sub_lst4 =  [each for each in os.listdir(disp_dir_train_3)]
                         for pfm_file in sub_lst4: # '0006.pfm'~'0015.pfm'
                             whole_file_path = disp_dir_train_3 + '/' + pfm_file
                             test_disp_left_lst.append(whole_file_path)
                             # print(whole_file_path)
                 
             
    for img_dir in imag_folders:
        
        sub_lst1 = [each for each in os.listdir(img_dir)]
        for sub_dir1 in sub_lst1:
            if sub_dir1 == 'TRAIN':
                pass
            elif sub_dir1 == 'TEST':
                pass
            else:
                raise Exception('Do not mix irrelevant files in the dataset')
            
            img_dir_train_1 = img_dir+'/'+sub_dir1
            sub_lst2 = [each for each in os.listdir(img_dir_train_1)]
            for sub_dir2 in sub_lst2: # 'A' 'B' 'C'
                img_dir_train_2 = img_dir_train_1 + '/' + sub_dir2
                
                sub_lst3 = [each for each in os.listdir(img_dir_train_2)]
                for sub_dir3 in sub_lst3: # '0000' '0001' '0002' ~ '0749'
                    img_dir_train_3 = img_dir_train_2 + '/' + sub_dir3

                    img_dir_train_3_left = img_dir_train_3 + '/' + 'left'   
                    img_dir_train_3_right = img_dir_train_3 + '/' + 'right'
                    
                    sub_lst4 =  [each for each in os.listdir(img_dir_train_3_left)]
                    for img_file in sub_lst4: # '0006.pfm'~'0015.pfm'
                        whole_file_path = img_dir_train_3_left + '/' + img_file
                        if(sub_dir1 == 'TRAIN'):
                            train_img_left_lst.append(whole_file_path)
                        elif sub_dir1 == 'TEST':
                            test_img_left_lst.append(whole_file_path)
                        else:
                            raise Exception('Do not mix irrelevant files in the dataset')
                        #print(whole_file_path)
                        
                    sub_lst4 =  [each for each in os.listdir(img_dir_train_3_right)]
                    for img_file in sub_lst4: # '0006.pfm'~'0015.pfm'
                        whole_file_path = img_dir_train_3_right + '/' + img_file
                        if(sub_dir1 == 'TRAIN'):
                            train_img_right_lst.append(whole_file_path)
                        elif sub_dir1 == 'TEST':
                            test_img_right_lst.append(whole_file_path)
                        else:
                            raise Exception('Do not mix irrelevant files in the dataset')
    
    assert len(train_img_left_lst) == len(train_img_right_lst)
    assert len(train_img_left_lst) == len(train_disp_left_lst)
    print(len(train_img_left_lst),',',len(train_img_right_lst),',',len(train_disp_left_lst))
    
    assert len(test_img_left_lst) == len(test_img_right_lst)
    assert len(test_img_left_lst) == len(test_disp_left_lst)
    print(len(test_img_left_lst),',',len(test_img_right_lst),',',len(test_disp_left_lst))
    
    #==========================================================================
    
    return (train_img_left_lst, train_img_right_lst, train_disp_left_lst, \
            test_img_left_lst, test_img_right_lst, test_disp_left_lst)



class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, left_img_lst, right_img_lst, left_disp_lst):
        self.left_img_lst = left_img_lst
        self.right_img_lst = right_img_lst
        self.left_disp_lst = left_disp_lst
        assert len(self.left_img_lst) == len(self.right_img_lst)
        assert len(self.left_img_lst) == len(self.left_disp_lst)
    
    def __getitem__(self,index):
        img_L_addr = self.left_img_lst[index]
        img_R_addr = self.right_img_lst[index]
        disp_L_addr = self.left_disp_lst[index]
        
        img_L = rgbimg_loader(img_L_addr)
        img_R = rgbimg_loader(img_R_addr)
        # disp_L形状是(H,W)
        disp_L = disp_loader(disp_L_addr)
        disp_L = np.ascontiguousarray(disp_L, dtype=np.float32)
        disp_L = torch.from_numpy(disp_L)            # 转成tensor
        
        # RGB图需要先reshape为(3,H,W)，再归一化到[0,1]之间，再(x-0.5)/0.5缩放到[-1,1]之间
        lst = [transforms.ToTensor(),transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))]
        preprocess = transforms.Compose(lst)
        img_L = preprocess (img_L)
        img_R = preprocess (img_R)
        
        return img_L, img_R, disp_L

    def __len__(self):
        return len(self.left_img_lst)
            
    
    

def write_lst_into_txt(name, lst):
    with open(name,'w') as f1:
        for each in lst:
            f1.write(each+'\n')

def read_lst_from_txt(name):
    with open(name,'r') as f1:
        # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
        lst = f1.readlines()
    
    lst = [each.strip() for each in lst]
    # for each in lst:
    #     print(each)
    return lst
    




if __name__ == "__main__":

    # read_PFM(r'D:\迅雷下载\disparity\TRAIN\A\0000\left\0006.pfm')
    
    
    
    ### 读取一边数据集地址后存在TXT中，以后就不需要重复遍历文件夹了
    train_img_left_lst, train_img_right_lst, train_disp_left_lst,\
    test_img_left_lst, test_img_right_lst, test_disp_left_lst \
        = list_all_img(r'D:\迅雷下载')
    
    write_lst_into_txt("train_img_left_lst.txt",train_img_left_lst)
    write_lst_into_txt("train_img_right_lst.txt",train_img_right_lst)
    write_lst_into_txt("train_disp_left_lst.txt",train_disp_left_lst)
    write_lst_into_txt("test_img_left_lst.txt",test_img_left_lst)
    write_lst_into_txt("test_img_right_lst.txt",test_img_right_lst)
    write_lst_into_txt("test_disp_left_lst.txt",test_disp_left_lst)
    
    
    
    ### 从TXT中读取得到数据集地址列表，不用再每次都花时间遍历文件夹了
    train_img_left_lst = read_lst_from_txt("train_img_left_lst.txt")
    train_img_right_lst = read_lst_from_txt("train_img_right_lst.txt")
    train_disp_left_lst = read_lst_from_txt("train_disp_left_lst.txt")
    test_img_left_lst = read_lst_from_txt("test_img_left_lst.txt")
    test_img_right_lst = read_lst_from_txt("test_img_right_lst.txt")
    test_disp_left_lst = read_lst_from_txt("test_disp_left_lst.txt")
    
    print(len(train_img_left_lst))
    print(len(train_img_right_lst))
    print(len(train_disp_left_lst))
    print(len(test_img_left_lst))
    print(len(test_img_right_lst))
    print(len(test_disp_left_lst))
    
    MyTrainDataset = MyDataset(train_img_left_lst, train_img_right_lst, train_disp_left_lst)
    MyTestDataset = MyDataset(test_img_left_lst, test_img_right_lst, test_disp_left_lst)
    
    # print(len(MyTrainDataset))
    # print(len(MyTestDataset))
    # x,y,z = MyTrainDataset[1]
    # print(x.shape,' ',y.shape,' ',z.shape)
    # print(type(x),' ',type(y),' ',type(z))
    # print('最大值：',x.max(),' 最小值：',x.min())
    # print('最大值：',y.max(),' 最小值：',y.min())
    # print('最大值：',z.max(),' 最小值：',z.min())
    
    # shuffle=True表示加载时打乱数据顺序
    # num_workers为用多少个子进程加载数据，0表示数据将在主进程中加载
    # 若数据集大小不能被batch__size整除，drop_last=True可删除最后一个不完整的batch
    # drop_last=False则最后一个batch将更小，默认为False
    
    
    
    TrainDataloader = torch.utils.data.DataLoader(MyTrainDataset,batch_size=1,
                                                  shuffle=True,num_workers=8,drop_last=False)
    TestDataloader = torch.utils.data.DataLoader(MyTestDataset,batch_size=1,
                                                  shuffle=True,num_workers=8,drop_last=False)
    
    for x,y,z in TrainDataloader:
        print(x.shape,",",y.shape,",",z.shape)


# input("END")
































