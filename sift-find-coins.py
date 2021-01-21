# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:16:23 2021

@author: 14044
"""

import cv2
import numpy as np

# 函数 根据特征点数找到合适的窗口大小
def find_window_size(biimg,minpts):
    # minpts=400
    print('finding window-size...')
    minpts+=1
    
    lcntpts=0
    lwindow=0
    
    # 获取点数
    height, width = biimg.shape
    ucntpts=0
    for i in range(height):
        for j in range(width):
            ucntpts+=biimg[i,j]
    
    
    # 计算初始窗口
    uwindow=min([height,width])
    window=int(uwindow/3)
    
    # window=int(np.sqrt(height*width*minpts/ucntpts))
    
    ucntpts=int(uwindow*uwindow/height/width*ucntpts)
    
    
    while True:
        # 用窗口遍历
        wcntpts=[0,width*height,0,0] #末两个数记录窗口位置
        print([lwindow,uwindow,window])
        
        for i in range(int(height/window)):
            for j in range(int(width/window)):
                wcntpts[0]=0
                for k1 in range(window):
                    for k2 in range(window):
                        wcntpts[0]+=biimg[i*window+k1,j*window+k2]
                if wcntpts[0]<wcntpts[1]:
                    wcntpts[1]=wcntpts[0]
                    wcntpts[2]=i*window
                    wcntpts[3]=j*window
        
        for i in range(int(height/window)):
            wcntpts[0]=0
            for k1 in range(window):
                    for k2 in range(window):
                        wcntpts[0]+=biimg[i*window+k1,width-window+k2]
            if wcntpts[0]<wcntpts[1]:
                wcntpts[1]=wcntpts[0]
                wcntpts[2]=i*window
                wcntpts[3]=width-window
        
        for j in range(int(width/window)):
            wcntpts[0]=0
            for k1 in range(window):
                for k2 in range(window):
                    wcntpts[0]+=biimg[height-window+k1,j*window+k2]
            if wcntpts[0]<wcntpts[1]:
                wcntpts[1]=wcntpts[0]
                wcntpts[2]=height-window
                wcntpts[3]=j*window
        
        wcntpts[0]=0        
        for k1 in range(window):
            for k2 in range(window):
                wcntpts[0]+=biimg[height-window+k1,width-window+k2]
        if wcntpts[0]<wcntpts[1]:
            wcntpts[1]=wcntpts[0]
            wcntpts[2]=height-window
            wcntpts[3]=width-window
        
    
        wcntpts[1]+=1
        
        if minpts-wcntpts[1]<0:
            ucntpts=wcntpts[1]
            uwindow=window
        elif minpts-wcntpts[1]>0:
            lcntpts=wcntpts[1]
            lwindow=window
        
        if uwindow<1.4*lwindow:
            print("window-size: %d" % window)
            return window
            # break; #程序结束，锁定窗口大小为window
        
        # if ucntpts-lcntpts==0:
        #     lcntpts=0
        #     lwindow=0
        
        
        window=int(np.sqrt((minpts-lcntpts)/(ucntpts-lcntpts)*(uwindow+lwindow)*(uwindow-lwindow)+lwindow*lwindow))
        
        if window==lwindow:
            print("window-size: %d" % window)
            return window
        #程序结束，锁定窗口大小为window
        
        if window>uwindow:
            print("window-size: %d" % uwindow)
            return uwindow
        #程序结束，找不到合适的窗口大小，只能暂定为uwindow


# 函数 判断特征点是否在窗内
def inwindow(keypoint,window_pos,window):
    if keypoint.pt[1]>=window_pos[0] and keypoint.pt[1]<window_pos[0]+window and keypoint.pt[0]>=window_pos[1] and keypoint.pt[0]<window_pos[1]+window:
        return True
    else:
        return False


imgl = cv2.imread('./1b.bmp')
imgtp = cv2.cvtColor(imgl, cv2.COLOR_RGB2GRAY) #模板图
src_range = np.float32([ [0,0],[0,imgtp.shape[1]-1],[imgtp.shape[0]-1,imgtp.shape[1]-1],[imgtp.shape[0]-1,0],[imgtp.shape[0]*0.5,imgtp.shape[1]*0.5] ]).reshape(-1,1,2)
_,tpmask=cv2.threshold(cv2.cvtColor(cv2.imread('./1tpmask.bmp'), cv2.COLOR_BGR2GRAY), 100, 1, cv2.THRESH_BINARY) #模板掩膜
tprange=cv2.cvtColor(cv2.imread('./1tprange.bmp'), cv2.COLOR_BGR2GRAY) #模板上的硬币范围，也可以看做一个掩膜
sift = cv2.xfeatures2d.SIFT_create(nfeatures=400,nOctaveLayers=4,contrastThreshold=0.09,edgeThreshold=8) #SIFT算子，限制400个特征点
# kpl,desl=sift.detectAndCompute(imgtp,mask=tpmask)
kpl,desl=sift.detectAndCompute(imgtp,mask=None) #检测模板上的SIFT特征点，计算每个特征点的特征值

minpts=len(kpl)

imgr = cv2.cvtColor(cv2.imread('./sample65.jpg'), cv2.COLOR_RGB2GRAY) #待检测图
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=4,contrastThreshold=0.09,edgeThreshold=8) #取消特征点数限制
kpr,desr=sift.detectAndCompute(imgr,None)

imgkpr=np.zeros(imgr.shape,dtype=np.uint8)
for keypoint in kpr:
    imgkpr[int(keypoint.pt[1]+0.5),int(keypoint.pt[0]+0.5)]=1

window=find_window_size(imgkpr, minpts) #使用的窗口大小

height,width=imgkpr.shape #待检测图的宽和高

coin_cnt=0 #计硬币个数
mask=np.ones(len(kpr),dtype=np.uint8) #特征点掩膜

#创建bf对象，并设定初始值
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

lock_flag=False #窗口是否已固定，固定时 lock_flag 为 True
lock_window=0 #确定找到一枚硬币后，可以将窗口固定下来，这个时候窗口值保存在 lock_window 中

while np.sum(mask)>0:
    new_window=height
    lock_flag_change=False
    for i in range(int(height/window)):
            for j in range(int(width/window)):
                #定位窗口到 i*window,j*window
                #寻找在窗口内的特征点，并且特征点处于启用状态
                ikpr_in=[kp_idx for kp_idx in range(len(kpr)) if mask[kp_idx]==1 and inwindow(kpr[kp_idx],[i*window,j*window],window)]
                if len(ikpr_in)==0: #如果窗口内的所有特征点都被关闭了，就不用匹配了
                    if new_window<height and not lock_flag:
                        window=int(new_window) #换成新窗口，准备再次搜索
                    continue
                kpr_in=[kpr[kp_idx] for kp_idx in ikpr_in]
                desr_in=desr[ikpr_in,:]
    
                #进行暴力匹配
                
                matches = bf.match(desl, desr_in)
                
                #将匹配结果按特征点之间的距离进行降序排列
                matches = sorted(matches, key= lambda x:x.distance)
                
                #距离均值需满足条件
                #最短的前60个距离需满足条件
                dis_mean=0
                dis_th_pos=60
                for i2 in range(len(matches)):
                    dis_mean+=matches[i2].distance
                for i2 in range(len(matches)):
                    if matches[i2].distance>302:
                        dis_th_pos=i2
                        break
                dis_mean=dis_mean/len(matches)
                #如果距离在范围内，计算新的窗口大小
                if dis_mean<345 and dis_th_pos>=60: ###
                    # 进行RANSAC
                    src_pts = np.float32([kpl[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst_pts = np.float32([ kpr_in[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                    M, quality = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=10.0,maxIters=6000,confidence=0.99)
                    # M, quality = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS,maxIters=5000,confidence=0.95)
                    if np.sum(quality)>30:  ### 判断quality是否满足要求
                        coin_cnt+=1
                        lock_flag=True
                        lock_flag_change=True
                        dst_range = cv2.perspectiveTransform(src_range,M)
                        temp_window=max([np.sqrt((dst_range[0][0][0]-dst_range[2][0][0])**2+(dst_range[0][0][1]-dst_range[2][0][1])**2),np.sqrt((dst_range[1][0][0]-dst_range[3][0][0])**2+(dst_range[1][0][1]-dst_range[3][0][1])**2)])
                        temp_window/=np.sqrt(2)
                        lock_window=temp_window
                        # 确定找到一枚硬币，可以关闭投影范围内的特征点
                        inv_M=np.linalg.inv(M)
                        dst_pts_in = np.float32([ kp.pt for kp in kpr ]).reshape(-1,1,2)
                        src_pts_in=cv2.perspectiveTransform(dst_pts_in,inv_M)
                        for pt_idx in range(src_pts_in.shape[0]):
                            src_row=int(src_pts_in[pt_idx,0,1]+0.5)
                            src_col=int(src_pts_in[pt_idx,0,0]+0.5)
                            if src_row>=0 and src_row<tprange.shape[0] and src_col>=0 and src_col<tprange.shape[1] and tprange[src_row,src_col]==1:
                                mask[pt_idx]=0
                          
                    else: #quality不满足要求,不关闭窗口内特征点，因为均值和60位值满足条件，可能是窗口内有多个硬币
                        if lock_flag and dis_th_pos>90:
                            coin_cnt+=1
                            for kpr_idx in ikpr_in:
                                mask[kpr_idx]=0   #关闭mask
                        elif lock_flag:
                            for kpr_idx in ikpr_in:
                                mask[kpr_idx]=0   #关闭mask
                        else:
                            #先将窗口范围内的特征点截取出来
                            imgkpr_in=imgkpr[i*window:i*window+window,j*window:j*window+window]
                            #计算新窗口大小
                            temp_window=find_window_size(imgkpr_in,minpts)
                            if temp_window<new_window:
                                new_window=temp_window
            
                #如果距离不在范围内，
                else:
                    #如果最短的前60个距离大于阈值302，关闭整个区域
                    for kpr_idx in ikpr_in:
                        mask[kpr_idx]=0   #关闭mask

                    #如果最短的前60个距离没那么大，这个区域还可能有东西，只将超过阈值的特征点和没有匹配上的特征点变为关闭状态
                    if dis_th_pos>=60:
                        for i2 in range(dis_th_pos):
                            mask[ikpr_in[matches[i2].trainIdx]]=1   #距离阈值范围内的特征点变为开启状态
                        
    if lock_flag and lock_flag_change:
        window=int(lock_window+0.5)
        continue
    
    for i in range(int(height/window)):
        #定位窗口到 i*window,width-window
        #寻找在窗口内的特征点，并且特征点处于启用状态
        ikpr_in=[kp_idx for kp_idx in range(len(kpr)) if mask[kp_idx]==1 and inwindow(kpr[kp_idx],[i*window,width-window],window)]
        if len(ikpr_in)==0: #如果窗口内的所有特征点都被关闭了，就不用匹配了
            if new_window<height and not lock_flag:
                window=int(new_window) #换成新窗口，准备再次搜索
            continue
        kpr_in=[kpr[kp_idx] for kp_idx in ikpr_in]
        desr_in=desr[ikpr_in,:]
   
        #进行暴力匹配
        
        matches = bf.match(desl, desr_in)
        
        #将匹配结果按特征点之间的距离进行降序排列
        matches = sorted(matches, key= lambda x:x.distance)
        
       
        
        #距离均值需满足条件
        #最短的前60个距离需满足条件
        dis_mean=0
        dis_th_pos=60
        for i2 in range(len(matches)):
            dis_mean+=matches[i2].distance
        for i2 in range(len(matches)):
            if matches[i2].distance>302:
                dis_th_pos=i2
                break
        dis_mean=dis_mean/len(matches)
        #如果距离在范围内，计算新的窗口大小
        if dis_mean<345 and dis_th_pos>=60: ###
            # 进行RANSAC
            src_pts = np.float32([kpl[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([ kpr_in[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, quality = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=10.0,maxIters=6000,confidence=0.99)
            if np.sum(quality)>30:  ### 判断quality是否满足要求
                coin_cnt+=1
                lock_flag=True
                dst_range = cv2.perspectiveTransform(src_range,M)
                temp_window=max([np.sqrt((dst_range[0][0][0]-dst_range[2][0][0])**2+(dst_range[0][0][1]-dst_range[2][0][1])**2),np.sqrt((dst_range[1][0][0]-dst_range[3][0][0])**2+(dst_range[1][0][1]-dst_range[3][0][1])**2)])
                temp_window/=np.sqrt(2)
                
                lock_window=temp_window
                # 关闭投影范围内的特征点
                inv_M=np.linalg.inv(M)
                dst_pts_in = np.float32([ kp.pt for kp in kpr ]).reshape(-1,1,2)
                src_pts_in=cv2.perspectiveTransform(dst_pts_in,inv_M)
                for pt_idx in range(src_pts_in.shape[0]):
                    src_row=int(src_pts_in[pt_idx,0,1]+0.5)
                    src_col=int(src_pts_in[pt_idx,0,0]+0.5)
                    if src_row>=0 and src_row<tprange.shape[0] and src_col>=0 and src_col<tprange.shape[0] and tprange[src_row,src_col]==1:
                        mask[pt_idx]=0
                  
            else: #quality不满足要求,不关闭窗口内特征点，因为均值和60位值满足条件，可能是窗口内有多个硬币
                if lock_flag and dis_th_pos>90:
                    coin_cnt+=1
                    for kpr_idx in ikpr_in:
                        mask[kpr_idx]=0   #关闭mask
                elif lock_flag:
                    for kpr_idx in ikpr_in:
                        mask[kpr_idx]=0   #关闭mask
                else:
                    #先将窗口范围内的特征点截取出来
                    imgkpr_in=imgkpr[i*window:i*window+window,j*window:j*window+window]
                    #计算新窗口大小
                    temp_window=find_window_size(imgkpr_in,minpts)
                    if temp_window<new_window:
                        new_window=temp_window
    
        #如果距离不在范围内，
        else:
            #如果最短的前60个距离大于阈值302，关闭整个区域
            for kpr_idx in ikpr_in:
                mask[kpr_idx]=0   #关闭mask
            # if dis_th_pos<25:
            #     for i2 in range(dis_th_pos):
            #         mask[ikpr_in[matches[i2].trainIdx]]=0   #关闭mask
            #如果最短的前60个距离没那么大，这个区域还可能有东西，只将超过阈值的特征点和没有匹配上的特征点变为关闭状态
            if dis_th_pos>=60:
                for i2 in range(dis_th_pos):
                    mask[ikpr_in[matches[i2].trainIdx]]=1   #开启mask
                
    
    
    for j in range(int(width/window)):
        #定位窗口到 height-window,j*window
        #寻找在窗口内的特征点，并且特征点处于启用状态
        ikpr_in=[kp_idx for kp_idx in range(len(kpr)) if mask[kp_idx]==1 and inwindow(kpr[kp_idx],[height-window,j*window],window)]
        if len(ikpr_in)==0: #如果窗口内的所有特征点都被关闭了，就不用匹配了
            if new_window<height and not lock_flag:
                window=int(new_window) #换成新窗口，准备再次搜索    
            # if lock_flag:
            #     window=int(lock_window+0.5)
            continue
        kpr_in=[kpr[kp_idx] for kp_idx in ikpr_in]
        desr_in=desr[ikpr_in,:]
  
        #进行暴力匹配
        
        matches = bf.match(desl, desr_in)
        
        #将匹配结果按特征点之间的距离进行降序排列
        matches = sorted(matches, key= lambda x:x.distance)
        
        
        #距离均值需满足条件
        #最短的前60个距离需满足条件
        dis_mean=0
        dis_th_pos=60
        for i2 in range(len(matches)):
            dis_mean+=matches[i2].distance
        for i2 in range(len(matches)):
            if matches[i2].distance>302:
                dis_th_pos=i2
                break
        dis_mean=dis_mean/len(matches)
        #如果距离在范围内，计算新的窗口大小
        if dis_mean<345 and dis_th_pos>=60: ###
            # 进行RANSAC
            src_pts = np.float32([kpl[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([ kpr_in[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, quality = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=10.0,maxIters=6000,confidence=0.99)
            if np.sum(quality)>30:  ### 判断quality是否满足要求
                coin_cnt+=1
                lock_flag=True
                dst_range = cv2.perspectiveTransform(src_range,M)
                temp_window=max([np.sqrt((dst_range[0][0][0]-dst_range[2][0][0])**2+(dst_range[0][0][1]-dst_range[2][0][1])**2),np.sqrt((dst_range[1][0][0]-dst_range[3][0][0])**2+(dst_range[1][0][1]-dst_range[3][0][1])**2)])
                temp_window/=np.sqrt(2)
                # if temp_window<new_window:
                lock_window=temp_window
                # 关闭投影范围内的特征点
                inv_M=np.linalg.inv(M)
                dst_pts_in = np.float32([ kp.pt for kp in kpr ]).reshape(-1,1,2)
                src_pts_in=cv2.perspectiveTransform(dst_pts_in,inv_M)
                for pt_idx in range(src_pts_in.shape[0]):
                    src_row=int(src_pts_in[pt_idx,0,1]+0.5)
                    src_col=int(src_pts_in[pt_idx,0,0]+0.5)
                    if src_row>=0 and src_row<tprange.shape[0] and src_col>=0 and src_col<tprange.shape[0] and tprange[src_row,src_col]==1:
                        mask[pt_idx]=0
                  
            else: #quality不满足要求,不关闭窗口内特征点，因为均值和60位值满足条件，可能是窗口内有多个硬币
                if lock_flag and dis_th_pos>90:
                    coin_cnt+=1
                    for kpr_idx in ikpr_in:
                        mask[kpr_idx]=0   #关闭mask
                elif lock_flag:
                    for kpr_idx in ikpr_in:
                        mask[kpr_idx]=0   #关闭mask
                else:
                    #先将窗口范围内的特征点截取出来
                    imgkpr_in=imgkpr[i*window:i*window+window,j*window:j*window+window]
                    #计算新窗口大小
                    temp_window=find_window_size(imgkpr_in,minpts)
                    if temp_window<new_window:
                        new_window=temp_window
    
        #如果距离不在范围内，
        else:
            #如果最短的前60个距离大于阈值302，关闭整个区域
            for kpr_idx in ikpr_in:
                mask[kpr_idx]=0   #关闭mask
           
            #如果最短的前60个距离没那么大，这个区域还可能有东西，只将超过阈值的特征点和没有匹配上的特征点变为关闭状态
            if dis_th_pos>=60:
                for i2 in range(dis_th_pos):
                    mask[ikpr_in[matches[i2].trainIdx]]=1   #开启mask
                
         
                
    
    
    #定位窗口到 height-window,width-window
    #寻找在窗口内的特征点，并且特征点处于启用状态
    ikpr_in=[kp_idx for kp_idx in range(len(kpr)) if mask[kp_idx]==1 and inwindow(kpr[kp_idx],[height-window,width-window],window)]
    if len(ikpr_in)==0: #如果窗口内的所有特征点都被关闭了，就不用匹配了
        if new_window<height and not lock_flag:
            window=int(new_window) #换成新窗口，准备再次搜索
        if lock_flag:
                window=int(lock_window+0.5)
        continue
    kpr_in=[kpr[kp_idx] for kp_idx in ikpr_in]
    desr_in=desr[ikpr_in,:]

    #进行暴力匹配
    
    matches = bf.match(desl, desr_in)
    
    #将匹配结果按特征点之间的距离进行降序排列
    matches = sorted(matches, key= lambda x:x.distance)
    
  
    #距离均值需满足条件
    #最短的前60个距离需满足条件
    dis_mean=0
    dis_th_pos=60
    for i2 in range(len(matches)):
        dis_mean+=matches[i2].distance
    for i2 in range(len(matches)):
        if matches[i2].distance>302:
            dis_th_pos=i2
            break
    dis_mean=dis_mean/len(matches)
    #如果距离在范围内，计算新的窗口大小
    if dis_mean<345 and dis_th_pos>=60: ###
        # 进行RANSAC
        src_pts = np.float32([kpl[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([ kpr_in[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, quality = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=10.0,maxIters=6000,confidence=0.99)
        if np.sum(quality)>30:  ### 判断quality是否满足要求
            coin_cnt+=1
            lock_flag=True
            dst_range = cv2.perspectiveTransform(src_range,M)
            temp_window=max([np.sqrt((dst_range[0][0][0]-dst_range[2][0][0])**2+(dst_range[0][0][1]-dst_range[2][0][1])**2),np.sqrt((dst_range[1][0][0]-dst_range[3][0][0])**2+(dst_range[1][0][1]-dst_range[3][0][1])**2)])
            temp_window/=np.sqrt(2)
           
            lock_window=temp_window
            # 关闭投影范围内的特征点
            inv_M=np.linalg.inv(M)
            dst_pts_in = np.float32([ kp.pt for kp in kpr ]).reshape(-1,1,2)
            src_pts_in=cv2.perspectiveTransform(dst_pts_in,inv_M)
            for pt_idx in range(src_pts_in.shape[0]):
                src_row=int(src_pts_in[pt_idx,0,1]+0.5)
                src_col=int(src_pts_in[pt_idx,0,0]+0.5)
                if src_row>=0 and src_row<tprange.shape[0] and src_col>=0 and src_col<tprange.shape[0] and tprange[src_row,src_col]==1:
                    mask[pt_idx]=0
              
        else: #quality不满足要求,不关闭窗口内特征点，因为均值和60位值满足条件，可能是窗口内有多个硬币
            if lock_flag and dis_th_pos>90:
                coin_cnt+=1
                for kpr_idx in ikpr_in:
                    mask[kpr_idx]=0   #关闭mask
            elif lock_flag:
                for kpr_idx in ikpr_in:
                    mask[kpr_idx]=0   #关闭mask
            else:
                #先将窗口范围内的特征点截取出来
                imgkpr_in=imgkpr[i*window:i*window+window,j*window:j*window+window]
                #计算新窗口大小
                temp_window=find_window_size(imgkpr_in,minpts)
                if temp_window<new_window:
                    new_window=temp_window
    
    #如果距离不在范围内，
    else:
        #如果最短的前60个距离大于阈值302，关闭整个区域
            for kpr_idx in ikpr_in:
                mask[kpr_idx]=0   #关闭mask
    
            #如果最短的前60个距离没那么大，这个区域还可能有东西，只将超过阈值的特征点和没有匹配上的特征点变为关闭状态
            if dis_th_pos>=60:
                for i2 in range(dis_th_pos):
                    mask[ikpr_in[matches[i2].trainIdx]]=1   #开启mask
            
    
    if new_window<height and not lock_flag:
        window=int(new_window) #换成新窗口，准备再次搜索
    if lock_flag:
        window=int(lock_window+0.5)
        
        
print("找到1块钱背面 %d 个" % coin_cnt)