#-*- coding:utf-8 -*-
# '''
# @Author: Hans Hu
# '''

import torch
import torch.nn as nn
import torch.nn.init as init
import math

class ssrnet(nn.Module):
    def __init__(self,stage_num,lambda_local,lambda_d,age):
        super(ssrnet, self).__init__()

        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d
        self.age = age

        # -------------------------------------------------
        self.x1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.x_layer1=nn.AvgPool2d(kernel_size=2,stride=2)

        self.x2=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.x_layer2=nn.AvgPool2d(kernel_size=2,stride=2) 

        self.x3=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.x_layer3=nn.AvgPool2d(kernel_size=2,stride=2) 

        self.x4=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # -------------------------------------------------
        self.s1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.s_layer1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.s2=nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.s_layer2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.s3=nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.s_layer3=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.s4=nn.Sequential(
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        # -------------------------------------------------

        # -------------------------------------------------
        self.s_layer4=nn.Sequential(
            nn.Conv2d(16,10,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
        )
        self.s_layer4_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(640,3),
            nn.ReLU(inplace=True),
        )
        # -------------------------------------------------

        # -------------------------------------------------
        self.x_layer4=nn.Sequential(
            nn.Conv2d(32,10,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
        )
        self.x_layer4_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(640,3),
            nn.ReLU(inplace=True),
        )
        # -------------------------------------------------        

        # -------------------------------------------------        
        self.delta_s1=nn.Sequential(
            nn.Linear(640,1),
            nn.Tanh(),
        )
        # -------------------------------------------------        

        # ------------------------------------------------- 
        self.feat_a_s1=nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(inplace=True),
        )       
        self.pred_a_s1=nn.Sequential(
            nn.Linear(6,3),
            nn.ReLU(inplace=True),
        )
        self.local_s1=nn.Sequential(
            nn.Linear(6,3),
            nn.Tanh(),
        )
        # -------------------------------------------------        

        # -------------------------------------------------        
        self.s_layer2_2=nn.Sequential(
            nn.Conv2d(16,10,kernel_size=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4,stride=4),
        )
        self.s_layer2_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(160,3),
            nn.ReLU(inplace=True),
        )
        # -------------------------------------------------        

        # -------------------------------------------------        
        self.x_layer2_2=nn.Sequential(
            nn.Conv2d(32,10,kernel_size=1,padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4,stride=4),
        )
        self.x_layer2_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(160,3),
            nn.ReLU(inplace=True),
        )
        # ------------------------------------------------- 

        # -------------------------------------------------        
        self.delta_s2=nn.Sequential(
            nn.Linear(160,1),
            nn.Tanh(),
        )
        # ------------------------------------------------- 

        # -------------------------------------------------        
        self.feat_a_s2=nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(inplace=True),
        )  
        self.pred_a_s2=nn.Sequential(
            nn.Linear(6,3),
            nn.ReLU(inplace=True),
        )
        self.local_s2=nn.Sequential(
            nn.Linear(6,3),
            nn.Tanh(),
        )
        # -------------------------------------------------

        # -------------------------------------------------        
        self.s_layer1_2=nn.Sequential(
            nn.Conv2d(16,10,kernel_size=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8,stride=8),
        )
        self.s_layer1_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(160,3),
            nn.ReLU(inplace=True),
        )
        # -------------------------------------------------        

        # -------------------------------------------------        
        self.x_layer1_2=nn.Sequential(
            nn.Conv2d(32,10,kernel_size=1,padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8,stride=8),
        )
        self.x_layer1_mix=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(160,3),
            nn.ReLU(inplace=True),
        )
        # ------------------------------------------------- 

        # -------------------------------------------------        
        self.delta_s3=nn.Sequential(
            nn.Linear(160,1),
            nn.Tanh(),
        )
        # ------------------------------------------------- 

        # -------------------------------------------------        
        self.feat_a_s3=nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(inplace=True),
        )  
        self.pred_a_s3=nn.Sequential(
            nn.Linear(6,3),
            nn.ReLU(inplace=True),
        )
        self.local_s3=nn.Sequential(
            nn.Linear(6,3),
            nn.Tanh(),
        )
        # -------------------------------------------------
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 0.5)
                init.constant(m.bias, 0.6)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.1)
                if m.bias is not None:
                    init.constant(m.bias, 0.3)

    def forward(self,x):
        # x->x1->x_layer1 : 3*64*64->32*32*32
        x1=self.x1(x)           
        x_layer1=self.x_layer1(x1)

        # x_layer1->x2->x_layer2 : 32*32*32->32*16*16
        x2=self.x2(x_layer1)
        x_layer2=self.x_layer2(x2)

        # x_layer2->x3->x_layer3->x4 : 32*16*16->32*8*8
        x3=self.x3(x_layer2)
        x_layer3=self.x_layer3(x3)
        x4=self.x4(x_layer3)

        # x->s1->s_layer1 : 3*64*64->16*32*32
        s1=self.s1(x)
        s_layer1=self.s_layer1(s1)

        # s_layer1->s2->s_layer2 : 16*32*32->16*16*16
        s2=self.s2(s_layer1)
        s_layer2=self.s_layer2(s2)

        # s_layer2->s3->s_layer3->s4 : 16*16*16->16*8*8
        s3=self.s3(s_layer2)
        s_layer3=self.s_layer3(s3)
        s4=self.s4(s_layer3)

        # s4->s_layer4->s_layer4_mix : 16*8*8->10*8*8->640->3
        s_layer4=self.s_layer4(s4)
        s_layer4=s_layer4.view(s_layer4.size(0), -1)
        s_layer4_mix=self.s_layer4_mix(s_layer4)

        # x4->x_layer4->x_layer4_mix : 32*8*8->10*8*8->640->3
        x_layer4=self.x_layer4(x4)
        x_layer4=x_layer4.view(x_layer4.size(0), -1)
        x_layer4_mix=self.x_layer4_mix(x_layer4)

        # feat_a_s1_pre->delta_s1 : 640->1
        feat_a_s1_pre=s_layer4.mul(x_layer4)
        delta_s1=self.delta_s1(feat_a_s1_pre)

        # feat_a_s1->pred_a_s1 : 3->6->3
        # feat_a_s1->local_s1 : 3->6->3
        feat_a_s1=s_layer4_mix.mul(x_layer4_mix)
        feat_a_s1=self.feat_a_s1(feat_a_s1)
        pred_a_s1=self.pred_a_s1(feat_a_s1)
        local_s1=self.local_s1(feat_a_s1)

        # s_layer2->s_layer2_mix : 16*16*16->10*16*16->10*4*4->160->3
        s_layer2=self.s_layer2_2(s_layer2)
        s_layer2=s_layer2.view(s_layer2.size(0), -1)
        s_layer2_mix=self.s_layer2_mix(s_layer2)

        # x_layer2->x_layer2_mix : 32*16*16->10*16*16->10*4*4->160->3
        x_layer2=self.x_layer2_2(x_layer2)
        x_layer2=x_layer2.view(x_layer2.size(0), -1)
        x_layer2_mix=self.x_layer2_mix(x_layer2)

        # feat_a_s2_pre->delta_s2 : 160->1
        feat_a_s2_pre=s_layer2.mul(x_layer2)
        delta_s2=self.delta_s2(feat_a_s2_pre)

        # feat_a_s2->pred_a_s2 : 3->6->3
        # feat_a_s2->local_s2 : 3->6->3
        feat_a_s2=s_layer2_mix.mul(x_layer2_mix)
        feat_a_s2=self.feat_a_s2(feat_a_s2)
        pred_a_s2=self.pred_a_s2(feat_a_s2)
        local_s2=self.local_s2(feat_a_s2)

        # s_layer1->s_layer1_mix : 16*32*32->10*32*32->10*4*4->160->3
        s_layer1=self.s_layer1_2(s_layer1)
        s_layer1=s_layer1.view(s_layer1.size(0), -1)
        s_layer1_mix=self.s_layer1_mix(s_layer1)

        # x_layer1->x_layer1_mix : 32*32*32->10*32*32->10*4*4->160->3
        x_layer1=self.x_layer1_2(x_layer1)
        x_layer1=x_layer1.view(x_layer1.size(0), -1)
        x_layer1_mix=self.x_layer1_mix(x_layer1)

        # feat_a_s3_pre->delta_s3 : 160->1
        feat_a_s3_pre=s_layer1.mul(x_layer1)
        delta_s3=self.delta_s3(feat_a_s3_pre)

        # feat_a_s3->pred_a_s3 : 3->6->3
        # feat_a_s3->local_s3 : 3->6->3
        feat_a_s3=s_layer1_mix.mul(x_layer1_mix)
        feat_a_s3=self.feat_a_s3(feat_a_s3)
        pred_a_s3=self.pred_a_s3(feat_a_s3)
        local_s3=self.local_s3(feat_a_s3)

        a = pred_a_s1[:,0]*0
        b = pred_a_s1[:,0]*0
        c = pred_a_s1[:,0]*0

        for i in range(0,self.stage_num[0]):
            a = a+(i+self.lambda_local*local_s1[:,i])*pred_a_s1[:,i]
        a = torch.unsqueeze(a, 1)
        a = a / (self.stage_num[0] * (1 + self.lambda_d * delta_s1))

        for j in range(0,self.stage_num[1]):
            b = b+(j+self.lambda_local*local_s2[:,j])*pred_a_s2[:,j]
        b = torch.unsqueeze(b, 1)
        b = b / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (
        self.stage_num[1] * (1 + self.lambda_d * delta_s2))

        for k in range(0,self.stage_num[2]):
            c = c+(k+self.lambda_local*local_s3[:,k])*pred_a_s3[:,k]

        c = torch.unsqueeze(c,1)
        c = c / (self.stage_num[0] * (1 + self.lambda_d * delta_s1)) / (
        self.stage_num[1] * (1 + self.lambda_d * delta_s2)) / (
            self.stage_num[2] * (1 + self.lambda_d * delta_s3))

        if self.age:
            V=101
        else:
            V=1
        age = (a+b+c)*V
        age = torch.squeeze(age,1)
        return age
