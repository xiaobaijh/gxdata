# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:04:18 2021

@author: LSY
"""
from PIL import Image
from functools import reduce
import cv2
import time
from scipy.spatial.distance import pdist
import numpy as np
import datetime

def doubleCheckTimeOut(times: list, thd_upper: int, thd_lower: int):
    time_out_id = []
    for time in times:
        if (datetime.datetime.now() - time).seconds > thd_upper:
            for i, time in enumerate(times):
                if (datetime.datetime.now() - time).seconds > thd_lower:
                    time_out_id.append(i)
            break
    return time_out_id

def cal_dist(image1, image2, Normal = 400):
    img0 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if Normal != 0:
        img0 = cv2.resize(img0, (Normal, Normal))
        img1 = cv2.resize(img1, (Normal, Normal))
    img_v0 = img0.flatten()
    img_v1 = img1.flatten()
    X = np.vstack([img_v0, img_v1])
    return int(pdist(X, 'euclidean')[0])


#
# # 喂入图片，img0是yolo检测到的，img0_ref是截取的参考背景图像,img1和img1_ref是相隔一定时间后截图到同一位置的图像
# def smoke_or_not(img0, img1, img0_ref, img1_ref, thre=2000):
#     is_smoke = False
#     img_distance = is_imgs_similar(img0, img1)
#     img_ref__distance = is_imgs_similar(img0_ref, img1_ref)
#     if (img_distance - img_ref__distance) > thre:
#         is_smoke = True
#     return is_smoke


def doubleCheck(img0, img1, xyxys, ref_xyxys, score_add, score_minus, thd=2000):
    score = [0] * len(xyxys)

    # ref_xyxys = find_boxes(img0, xyxys)
    # rect = maxmalRectangle((w, h), xyxys)
    # ref_xyxys = findboxes4boxes(xyxys, rect)
    dis_list = []
    for i, xyxy in enumerate(xyxys):
        x0 = xyxy[0]
        y0 = xyxy[1]
        x1 = xyxy[2]
        y1 = xyxy[3]
## xyxys 和 ref_xyxys 的索引范围不一样。
        print('***debug*** ref_xyxys[i]', ref_xyxys[i])
        ref_x0 = int(ref_xyxys[i][0])
        ref_y0 = int(ref_xyxys[i][1])
        ref_x1 = int(ref_xyxys[i][2])
        ref_y1 = int(ref_xyxys[i][3])

        # debug
        img0_plot = img0.copy()
        img1_plot = img1.copy()
        cv2.namedWindow('img0_plot', cv2.WINDOW_FREERATIO)
        cv2.namedWindow('img1_plot', cv2.WINDOW_FREERATIO)
        cv2.rectangle(img0_plot, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.rectangle(img0_plot, (ref_x0, ref_y0), (ref_x1, ref_y1), (0, 255, 0), 3)
        cv2.rectangle(img1_plot, (x0, y0), (x1, y1), (255, 0, 0), 3)
        cv2.rectangle(img1_plot, (ref_x0, ref_y0), (ref_x1, ref_y1), (0, 255, 0), 3)
        cv2.imshow('img0_plot', img0_plot)
        cv2.imshow('img1_plot', img1_plot)
        cv2.waitKey(1)
        time.sleep(1)

        # debug
        # print('xyxys', xyxys)
        # print('debug_ref_img0_xyxy=', ref_y0, ref_y1, ref_x0, ref_x1)

        # cv2.imshow('dist', img0[y0:y1, x0:x1])
        # cv2.waitKey(0)
        img_distance = cal_dist(img0[y0:y1, x0:x1], img1[y0:y1, x0:x1])
        img_ref_distance = cal_dist(img0[ref_y0:ref_y1, ref_x0: ref_x1], img1[ref_y0:ref_y1, ref_x0: ref_x1])

        print('box', str(i) + '\'s dist=', img_distance - img_ref_distance)
        if (img_distance - img_ref_distance) > thd:
            score[i] = score_add
            # print('is smoke')
        else:
            score[i] = -1 * score_minus
            # print('not smoke')
        dis_list.append(img_distance - img_ref_distance)
    return score, dis_list
