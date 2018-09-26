
# coding: utf-8

# In[23]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

print("画像54枚にマスク画像を合成するよ！")
for i in range(1, 55):
    "画像の読み込み"
    "背景は、PRADAの服、前景はマスク画像(透過で読み込み)"  
#     src = cv2.imread("/Users/MorikawaLab/Pictures/PRADA_mask_var20180131.png", -1)
#     dst = cv2.imread("/Users/MorikawaLab/Desktop/PRADA/PRADA ({0:d}).jpg".format(i), -1)
    src = cv2.imread("/Users/facul/Downloads/PRADA_mask_haba55_up35_down50.png", -1) #PRADAで色相に関する最適なマスｋ画像
    dst = cv2.imread("/Users/facul/PRADA/PRADA ({0:d}).jpg".format(i), -1)
    dst_=dst.astype(np.float64)
    
    "前景の画像の大きさを調べる。"
    width = src.shape[0]
    height = src.shape[1]

    "マスク画像の作成"
    mask = src[:,:,3]   #これでアルファチャンネルのみの行列が抽出。配列のスライスの参考→https://qiita.com/okkn/items/54e81346d8f35733ab5e
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  #maskを3色分にした。0 ->(B0, G0, R0)になる。
    mask = mask/250   #正規化

    "画像の合成"
    src = src[:,:, :3]
    dst_ *= 1 - mask   # 透過率に応じて元の画像を暗くする。
    # dst_[:,:, :3] = (1 - mask) * dst_[:,:, :3]
    dst_ += src * mask
    # cv2.imwrite("/Users/MorikawaLab/Desktop/PRADA_gousei/PRADA_gousei ({0:d}).png".format(i), dst_)
    cv2.imwrite("/Users/facul/Pictures/20180814_PRADA_masked/PRADA_masked ({0:d}).png".format(i), dst_)


print("------------------------------------------------------------------")

print("これから画像の画素の操作をします")

for i in range(1, 55):
#     dst_ = cv2.imread("/Users/MorikawaLab/Desktop/PRADA_gousei/PRADA_gousei ({0:d}).png".format(i), -1)
    dst_ = cv2.imread("/Users/facul/Pictures/20180814_PRADA_masked/PRADA_masked ({0:d}).png".format(i), -1)

#     print(i,"番目の画像の処理をします。")
#     切り抜いた画像の色相分布について
#     その後、色相・彩度・明度の分布を行列化。  

    height, width = dst_.shape[ :2 ]
    
    black = 0
        
    for j in range(0, height):
        for k in range(0, width):
            
            "GIMPの提示する黒は(0, 0 ,0)なのでマスク部分をカウント"
            if dst_.item(j, k, 0)==0 and dst_.item(j, k, 1)==0 and dst_.item(j, k, 2)==0:
                black = black + 1
                
#     print(i, "番目の黒色の数は",black )
    
    img_after =  cv2.cvtColor(dst_, cv2.COLOR_BGR2HSV)
    hist_m0= cv2.calcHist([img_after], [0], None, [180], [0, 180]) 
#     np.savetxt('/Users/MorikawaLab/Desktop/PRADA_test/PRADAtest({0:d}).csv'.format(i),hist_m0, delimiter=',')
#     np.savetxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADAtest({0:d}).csv'.format(i),hist_m0, delimiter=',')
    hist_m0[0] -= black
#     np.savetxt('/Users/MorikawaLab/Desktop/PRADA_masked_20180206/PRADA_masked_hue/PRADA_masked_hue({0:d}).csv'.format(i), hist_m0, delimiter=',')
    np.savetxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_hue({0:d}).csv'.format(i),hist_m0, delimiter=',')

    hist_m1 = cv2.calcHist([img_after], [1], None, [256], [0, 256]) 
    hist_m1[0] -= black
    np.savetxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_sat({0:d}).csv'.format(i),hist_m1, delimiter=',')

    hist_m2 = cv2.calcHist([img_after], [2], None, [256], [0,256])
    hist_m2[0] -=  black
    np.savetxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_val({0:d}).csv'.format(i),hist_m2, delimiter=',')
    

print("マスク画像の色相・彩度・明度の行列化が終わりました")


print("教師データの行列データを読み込み、区分分けして、割合の式にあてはめ、一致率を算出")

correct_hue_total = 0
correct_sat_total = 0
correct_val_total = 0

for i in range(1, 4):
#     data1 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_masked_20180206/PRADA_masked_hue/PRADA_masked_hue({0:d}).csv'.format(i), delimiter=",")
    data1 = np.loadtxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_hue({0:d}).csv'.format(i), delimiter=",")
#     data2 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_teach_data/PRADA_teach_hue/PRADA_teach_hue({0:d}).csv'.format(i), delimiter=",")
    
    print(i)
    hue_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hue10 = (data1[171: 180].sum() + data1[2: 10].sum()) / (171 * 256 - black)
    hue_list[0] = hue10
    hue11 = data1[10:27].sum()  / (171 * 256 - black)
    hue_list[1] = hue11
    hue12 = data1[27: 45].sum() /  (171 * 256 - black)
    hue_list[2] = hue12
    hue13 = data1[45: 63].sum() /  (171 * 256 - black)
    hue_list[3] = hue13
    hue14 = data1[63: 81].sum() /  (171 * 256 - black)
    hue_list[4] = hue14
    hue15 = data1[81: 99].sum() /  (171 * 256 - black)
    hue_list[5] = hue15
    hue16 = data1[99: 117].sum() /  (171 * 256 - black)
    hue_list[6] = hue16
    hue17 = data1[117: 135].sum() /  (171 * 256 - black)
    hue_list[7] = hue17
    hue18 = data1[135: 153].sum() /  (171 * 256 - black)
    hue_list[8] = hue18
    hue19 = data1[153: 171].sum() /  (171 * 256 - black)
    hue_list[9] = hue19
    
    print(hue_list)
    
#     hue20 = (data2[171: 180].sum() + data2[2: 10].sum()) / (171 * 256 - black)
#     hue21 = data2[10:27].sum()  / (171 * 256 - black)
#     hue22 = data2[27: 45].sum() /  (171 * 256 - black)
#     hue23 = data2[45: 63].sum() /  (171 * 256 - black)
#     hue24 = data2[63: 81].sum() /  (171 * 256 - black)
#     hue25 = data2[81: 99].sum() /  (171 * 256 - black)
#     hue26 = data2[99: 117].sum() /  (171 * 256 - black)
#     hue27 = data2[117: 135].sum() /  (171 * 256 - black)
#     hue28 = data2[135: 153].sum() /  (171 * 256 - black)
#     hue29 = data2[153: 171].sum() /  (171 * 256 - black)
    
#     if hue10 ==0 and  hue20==0:  #マスク・教師ともに割合が0なら計算式がおかしくなるので先に一致率を1にする.
#         correct0 ==1
#     else:
#         correct0 = (1 - abs(hue10 - hue20)) / max(hue10, hue20)   #
    
#     if hue11 ==0 and  hue21==0:
#         correct1 ==1
#     else:
#         correct1 = (1 - abs(hue11 - hue21)) / max(hue11, hue21)
        
#     if hue12 ==0 and  hue22==0:
#         correct2 ==1
#     else:
#         correct2 = (1 - abs(hue12 - hue22)) / max(hue12, hue22)

#     if hue13 ==0 and  hue23 ==0:
#         correct3 ==1
#     else:
#         correct3 = (1 - abs(hue13 - hue23)) / max(hue13, hue23)
        
#     if hue14 ==0  and  hue24 ==0:
#         correct4 ==1
#     else:
#         correct4 = (1 - abs(hue14 - hue24)) / max(hue14, hue24)
    
#     if hue15 ==0 and  hue25 ==0:
#         correct5 ==1
#     else:
#         correct5 = (1 - abs(hue15 - hue25)) / max(hue15, hue25)
        
#     if hue16 ==0 and  hue26==0:
#         correct6 ==1
#     else:
#         correct6 = (1 - abs(hue16 - hue26)) / max(hue16, hue26)

#     if hue17 ==0 and  hue27 ==0:
#         correct7 ==1
#     else:
#         correct7 = 1 - abs(hue17 - hue27) / max(hue17, hue27)
        
#     if hue18 ==0 and  hue28==0:
#         correct8 ==1
#     else:
#         correct8 = (1 - abs(hue18 - hue28)) / max(hue18, hue28)

#     if hue19 ==0 and  hue29 ==0:
#         correct9 ==1
#     else:
#         correct9 = (1 - abs(hue19 - hue29)) / max(hue19, hue29)
            
#     correct_hue = (correct1 + correct2 + correct3 + correct4 +correct5 + correct6 + correct7 + correct8 + correct9 ) / 10
#     print(correct_hue)  #1枚の画像に対する一致率の平均
    
#     correct_hue_total += correct_hue

    # correct_hue_average = correct_hue_total /54   #54枚の画像に対する一致率の平均

    # print("色相の一致率は", correct_hue_average, "である")

#     print("--------------↑ここまで色相-----------↓ここから彩度--------------------------------------")


    
#     data1 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_masked_20180206/PRADA_masked_sat/PRADA_masked_sat({0:d}).csv'.format(i), delimiter=",")
    data1 = np.loadtxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_sat({0:d}).csv'.format(i), delimiter=",")
#     data2 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_teach_data/PRADA_teach_sat/PRADA_teach_sat({0:d}).csv'.format(i), delimiter=",")
    

    hue_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hue10 = (data1[171: 180].sum() + data1[2: 10].sum()) / (171 * 256 - black)
    hue_list[0] = hue10
    hue11 = data1[10:27].sum()  / (171 * 256 - black)
    hue_list[1] = hue11
    hue12 = data1[27: 45].sum() /  (171 * 256 - black)
    hue_list[2] = hue12
    hue13 = data1[45: 63].sum() /  (171 * 256 - black)
    hue_list[3] = hue13
    hue14 = data1[63: 81].sum() /  (171 * 256 - black)
    hue_list[4] = hue14
    hue15 = data1[81: 99].sum() /  (171 * 256 - black)
    hue_list[5] = hue15
    hue16 = data1[99: 117].sum() /  (171 * 256 - black)
    hue_list[6] = hue16
    hue17 = data1[117: 135].sum() /  (171 * 256 - black)
    hue_list[7] = hue17
    hue18 = data1[135: 153].sum() /  (171 * 256 - black)
    hue_list[8] = hue18
    hue19 = data1[153: 171].sum() /  (171 * 256 - black)
    hue_list[9] = hue19
    
    print(hue_list)
    
#     hue20 = (data2[171: 180].sum() + data2[2: 10].sum()) / (171 * 256 - black)
#     hue21 = data2[10:27].sum()  / (171 * 256 - black)
#     hue22 = data2[27: 45].sum() /  (171 * 256 - black)
#     hue23 = data2[45: 63].sum() /  (171 * 256 - black)
#     hue24 = data2[63: 81].sum() /  (171 * 256 - black)
#     hue25 = data2[81: 99].sum() /  (171 * 256 - black)
#     hue26 = data2[99: 117].sum() /  (171 * 256 - black)
#     hue27 = data2[117: 135].sum() /  (171 * 256 - black)
#     hue28 = data2[135: 153].sum() /  (171 * 256 - black)
#     hue29 = data2[153: 171].sum() /  (171 * 256 - black)
    
#     if hue10 ==0 and  hue20==0:  #マスク・教師ともに割合が0なら計算式がおかしくなるので先に一致率を1にする.
#         correct0 ==1
#     else:
#         correct0 = (1 - abs(hue10 - hue20)) / max(hue10, hue20)   #
    
#     if hue11 ==0 and  hue21==0:
#         correct1 ==1
#     else:
#         correct1 = (1 - abs(hue11 - hue21)) / max(hue11, hue21)
        
#     if hue12 ==0 and  hue22==0:
#         correct2 ==1
#     else:
#         correct2 = (1 - abs(hue12 - hue22)) / max(hue12, hue22)

#     if hue13 ==0 and  hue23 ==0:
#         correct3 ==1
#     else:
#         correct3 = (1 - abs(hue13 - hue23)) / max(hue13, hue23)
        
#     if hue14 ==0  and  hue24 ==0:
#         correct4 ==1
#     else:
#         correct4 = (1 - abs(hue14 - hue24)) / max(hue14, hue24)
    
#     if hue15 ==0 and  hue25 ==0:
#         correct5 ==1
#     else:
#         correct5 = (1 - abs(hue15 - hue25)) / max(hue15, hue25)
        
#     if hue16 ==0 and  hue26==0:
#         correct6 ==1
#     else:
#         correct6 = (1 - abs(hue16 - hue26)) / max(hue16, hue26)

#     if hue17 ==0 and  hue27 ==0:
#         correct7 ==1
#     else:
#         correct7 = 1 - abs(hue17 - hue27) / max(hue17, hue27)
        
#     if hue18 ==0 and  hue28==0:
#         correct8 ==1
#     else:
#         correct8 = (1 - abs(hue18 - hue28)) / max(hue18, hue28)

#     if hue19 ==0 and  hue29 ==0:
#         correct9 ==1
#     else:
#         correct9 = (1 - abs(hue19 - hue29)) / max(hue19, hue29)
            
#     correct_sat = (correct1 + correct2 + correct3 + correct4 +correct5 + correct6 + correct7 + correct8 + correct9 ) / 10
#     print(correct_sat)
    
#     correct_sat_total += correct_sat

# correct_sat_average = correct_sat_total /54

# print("彩度の一致率は", correct_sat_average, "である")

# print("---------------------↑ここまで彩度----↓これから明度--------------------------------")

# for i in range(1, 55):
#     data1 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_masked_20180206/PRADA_masked_val/PRADA_masked_val({0:d}).csv'.format(i), delimiter=",")
    data1 = np.loadtxt('/Users/facul/Pictures/20180814_PRADA_masked/PRADA_sat({0:d}).csv'.format(i), delimiter=",")
#     data2 = np.loadtxt('/Users/MorikawaLab/Desktop/PRADA_teach_data/PRADA_teach_val/PRADA_teach_val({0:d}).csv'.format(i), delimiter=",")
    
    hue_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    hue10 = (data1[171: 180].sum() + data1[2: 10].sum()) / (171 * 256 - black)
    hue_list[0] = hue10
    hue11 = data1[10:27].sum()  / (171 * 256 - black)
    hue_list[1] = hue11
    hue12 = data1[27: 45].sum() /  (171 * 256 - black)
    hue_list[2] = hue12
    hue13 = data1[45: 63].sum() /  (171 * 256 - black)
    hue_list[3] = hue13
    hue14 = data1[63: 81].sum() /  (171 * 256 - black)
    hue_list[4] = hue14
    hue15 = data1[81: 99].sum() /  (171 * 256 - black)
    hue_list[5] = hue15
    hue16 = data1[99: 117].sum() /  (171 * 256 - black)
    hue_list[6] = hue16
    hue17 = data1[117: 135].sum() /  (171 * 256 - black)
    hue_list[7] = hue17
    hue18 = data1[135: 153].sum() /  (171 * 256 - black)
    hue_list[8] = hue18
    hue19 = data1[153: 171].sum() /  (171 * 256 - black)
    hue_list[9] = hue19
    
    print(hue_list)
    
#     hue20 = (data2[171: 180].sum() + data2[2: 10].sum()) / (171 * 256 - black)
#     hue21 = data2[10:27].sum()  / (171 * 256 - black)
#     hue22 = data2[27: 45].sum() /  (171 * 256 - black)
#     hue23 = data2[45: 63].sum() /  (171 * 256 - black)
#     hue24 = data2[63: 81].sum() /  (171 * 256 - black)
#     hue25 = data2[81: 99].sum() /  (171 * 256 - black)
#     hue26 = data2[99: 117].sum() /  (171 * 256 - black)
#     hue27 = data2[117: 135].sum() /  (171 * 256 - black)
#     hue28 = data2[135: 153].sum() /  (171 * 256 - black)
#     hue29 = data2[153: 171].sum() /  (171 * 256 - black)
    
#     if hue10 ==0 and  hue20==0:  #マスク・教師ともに割合が0なら計算式がおかしくなるので先に一致率を1にする.
#         correct0 ==1
#     else:
#         correct0 = (1 - abs(hue10 - hue20)) / max(hue10, hue20)   #
    
#     if hue11 ==0 and  hue21==0:
#         correct1 ==1
#     else:
#         correct1 = (1 - abs(hue11 - hue21)) / max(hue11, hue21)
        
#     if hue12 ==0 and  hue22==0:
#         correct2 ==1
#     else:
#         correct2 = (1 - abs(hue12 - hue22)) / max(hue12, hue22)

#     if hue13 ==0 and  hue23 ==0:
#         correct3 ==1
#     else:
#         correct3 = (1 - abs(hue13 - hue23)) / max(hue13, hue23)
        
#     if hue14 ==0  and  hue24 ==0:
#         correct4 ==1
#     else:
#         correct4 = (1 - abs(hue14 - hue24)) / max(hue14, hue24)
    
#     if hue15 ==0 and  hue25 ==0:
#         correct5 ==1
#     else:
#         correct5 = (1 - abs(hue15 - hue25)) / max(hue15, hue25)
        
#     if hue16 ==0 and  hue26==0:
#         correct6 ==1
#     else:
#         correct6 = (1 - abs(hue16 - hue26)) / max(hue16, hue26)

#     if hue17 ==0 and  hue27 ==0:
#         correct7 ==1
#     else:
#         correct7 = 1 - abs(hue17 - hue27) / max(hue17, hue27)
        
#     if hue18 ==0 and  hue28==0:
#         correct8 ==1
#     else:
#         correct8 = (1 - abs(hue18 - hue28)) / max(hue18, hue28)

#     if hue19 ==0 and  hue29 ==0:
#         correct9 ==1
#     else:
#         correct9 = (1 - abs(hue19 - hue29)) / max(hue19, hue29)
            
#     correct_val = (correct1 + correct2 + correct3 + correct4 +correct5 + correct6 + correct7 + correct8 + correct9 ) / 10
#     print(correct_val)
    
#     correct_val_total += correct_val

    # correct_val_average = correct_val_total /54

    # print("明度の一致率は", correct_val_average, "である")
    
    print("-----------------------------------------------------------------------------")

