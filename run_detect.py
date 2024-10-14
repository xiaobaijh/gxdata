# coding: utf-8
# base import
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import datetime
import multiprocessing as mp
import socket
import json

# yolo import
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox

# alex import
from threading import Thread
from send_alert import sendPwr, sendAlert, loadStaff, sendRunning
from utils2 import calIou, imgs2video, loadConfig, nms, writeLog, findRed, removeFew
from double_check_1 import doubleCheck, doubleCheckTimeOut
from onvif_cam import OnvifHik
from oth.find_box_3 import findboxes4boxes, maxmalRectangle
from tcp_pic import pic_client

# aniie import
import logging
import configparser


class stringFilter(logging.Filter):
    def filter(self, record):
        if record.msg.find('Parameter') == -1:
            return True
        return False


# 通过下面的方式进行简单配置输出方式与日志级别
def init_log():
    log_data = datetime.datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.join('/home/fzb/storage/data', log_data)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print("**20")
        print(log_path)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    # log -> detect
    handler_detect = logging.FileHandler(os.path.join(log_path, "detect_log.txt"))
    handler_detect.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler_detect.setFormatter(formatter)
    logger.addHandler(handler_detect)
    return logger, log_data


def log_parameter():
    log_data = datetime.datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.join('/home/fzb/storage/data', log_data)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print("**20")
        print(log_path)

    logger_p = logging.getLogger("logger_p")
    logger_p.setLevel(logging.INFO)
    # log -> parameter_log.txt
    handler_detect = logging.FileHandler(os.path.join(log_path, "parameter_log.txt"))
    handler_detect.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler_detect.setFormatter(formatter)
    logger_p.addHandler(handler_detect)

    logger_p.info("first_check_stable_wait_time = %d" % first_check_stable_wait_time)
    logger_p.info("first_check_stable_wait_time_focal = %d" % first_check_stable_wait_time_focal)

    logger_p.info("first_check_detect_time_short = %d" % first_check_detect_time_short)
    logger_p.info("first_check_detect_time_long = %d" % first_check_detect_time_long)

    logger_p.info("min_red_pixel_count = %d" % min_red_pixel_count)

    logger_p.info("double_check_stable_wait_time = %d" % double_check_stable_wait_time)
    logger_p.info("double_check_detect_times = %d " % double_check_detect_times)
    logger_p.info("double_check_cal_times = %d" % double_check_cal_times)
    logger_p.info("double_check_interval_upper = %d", double_check_interval_upper)
    logger_p.info("double_check_interval_lower = %d", double_check_interval_lower)

    logger_p.info("double_check_score_add = %d" % double_check_score_add)
    logger_p.info("double_check_score_minus = %d" % double_check_score_minus)
    logger_p.info("double_check_score_thd = %d" % double_check_score_thd)

    # few remove
    logger_p.info("few_remove_count_thr = %d " % few_remove_count_thr)
    logger_p.info("few_remove_count_thr2 = %d" % few_remove_count_thr2)
    logger_p.info("few_remove_iou_thr = %f" % few_remove_iou_thr)

    # power thresholds
    logger_p.info("pwr_thr_lo = %d " % pwr_thr_lo)
    logger_p.info("pwr_thr_hi = %d " % pwr_thr_hi)

    logger_p.info("time_early_H = %d" % time_early_H)
    logger_p.info("time_early_M = %d" % time_early_M)
    logger_p.info("time_late_H = %d" % time_late_H)
    logger_p.info("time_late_M =%d" % time_late_M)

    logger_p.info("time_early = %d" % time_early)
    logger_p.info("time_late = %d " % time_late)
    logger_p.info("imgsz = %d" % imgsz)
    logger_p.info("conf_thres = %d" % conf_thres)
    logger_p.info("iou_thres = %d" % iou_thres)
    logger_p.info("min_xyxy_size = %d" % min_xyxy_size)
    logger_p.info("show_img = %s" % str(show_img))
    logger_p.info("show_print = %s" % str(show_print))
    logger_p.info("show_print_v =  %s" % str(show_print_v))
    logger_p.info("auto_resume_time = %d" % auto_resume_time)
    logger_p.info("send_running_interval = %s " % str(send_running_interval))
    logger_p.info("ContorlCenter_f = %s " % str(ContorlCenter_f))
    logger_p.info("config = %s" % str(config))
    logger_p.info("root_dir = %s" % str(root_dir))

    # camera
    logger_p.info("onvif_ip = %s" % str(onvif_ip))
    logger_p.info("onvif_user = %s" % str(onvif_user))
    logger_p.info("onvif_pwd = %s" % str(onvif_pwd))
    logger_p.info("cc_host = %s" % str(cc_host))
    logger_p.info("cc_port = %s" % str(cc_port))
    logger_p.info("pic_port = %s" % str(pic_port))
    logger_p.info("dev_id = %s" % str(dev_id))

    logger_p.info("north_angle = %s" % str(north_angle))
    logger_p.info("optical_path = %s" % str(optical_path))
    logger_p.info("thermal_path = %s" % str(thermal_path))
    # main
    logger_p.info("loc_cn = %s" % str(loc_cn))

    # files
    logger_p.info("files_dir = %s" % str(files_dir))
    logger_p.info("pos_list_file = %s" % str(pos_list_file))
    logger_p.info("pos_list_file = %s" % str(pos_list_file))
    logger_p.info("pos_list_file = %s" % str(pos_list_file))

    logger_p.info("staff_list_file = %s" % str(staff_list_file))

    logger_p.info("staff_filename = %s" % str(staff_filename))

    logger_p.info("weights_dir = %s" % str(weights_dir))
    logger_p.info("weight_filepath = %s" % str(weight_filepath))

    # path
    logger_p.info("save_path = %s" % str(save_path))

    logger_p.info("mask_filepath = %s" % str(mask_filepath))
    # para
    logger_p.info("doucheck_iou_thd = %s" % str(doucheck_iou_thd))
    logger_p.info("double_check_dist_thr = %s" % str(double_check_dist_thr))
    # 2000 for extra high, 2800 for high, 5000 for medium, 8000 for low, 10000 for extra low


def removeHysterisis():
    is_moving = cam0.isMoving()
    while is_moving:
        is_moving = cam0.isMoving()
    cam0.continuousMove(-1.0, -1.0, 0, 0.8)
    cam0.continuousZoom(-1, 2)
    is_moving = cam0.isMoving()
    while is_moving:
        is_moving = cam0.isMoving()
    cam0.continuousMove(1.0, 1.0, 0, 0.4)
    cam0.continuousZoom(1, 2)
    is_moving = cam0.isMoving()
    time.sleep(1)
    while is_moving:
        is_moving = cam0.isMoving()


def alert(alert_queue):
    while True:
        time.sleep(1)
        if alert_queue.qsize() > 0:
            print(alert_queue.qsize() - 1, 'alert(s) awaiting')
            host, pic_port, loc_cn, direc, input_img, input_video = alert_queue.get()
            filename = loc_cn + direc + datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
            sendAlert(loc_cn, direc, input_img, input_video)
            pic_client(host, pic_port, [filename, cv2.resize(cv2.imread(input_img), (1280, 720))])


def listenControlCenter(host, port, dev_id, interrupt, interrupted):
    while True:
        time.sleep(10)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect((host, port))
        except Exception as e:
            if ContorlCenter_f:
                print('cannot connect with control center, re-connecting')
                writeLog('E', 'detect', 'listenControlCenter', 'cannot connect with control center' + str(e))
        else:
            msg_ = {"c": "reg", "a": {"device_id": str(dev_id)}}
            # msg = bytes('{}'.format(msg_),'utf-8')
            msg = json.dumps(msg_)

            # print(msg)
            client.send(msg.encode('utf-8'))
            line = '\n'
            client.send(line.encode('utf-8'))
            print('registered with control center')
            writeLog('I', 'detect', 'listenControlCenter', 'registered with control center')

            while True:
                try:
                    data = client.recv(1024).decode()
                except Exception as e:
                    print('disconnected with control center')
                    writeLog('E', 'detect', 'listenControlCenter', 'disconnected with control center' + str(e))
                    return 1
                if data:
                    # print('received data', data)
                    interrupt.value = 1
                    while not interrupted.value:
                        print('waiting for stop detecting, interrupt =', interrupt.value)
                        writeLog('I', 'detect', 'listenControlCenter', 'waiting for stop detecting')
                        time.sleep(1)
                    try:
                        data = data[:data.find('}') + 1]
                        data_dict = json.loads(data)
                    except Exception as e:
                        print('json decode failed: ', data)
                        writeLog('E', 'detect', 'listenControlCenter', 'json decode failed:' + str(data) + str(e))
                    else:
                        command = data_dict['command']
                        print('***debug*** recv:', command)
                        writeLog('I', 'detect', 'listenControlCenter', 'recv command:' + str(command))

                        if command == '0008 0001':
                            # up
                            cam0.continuousMove(y_speed=0.01, stop_time=0.5)
                        elif command == '0010 0001':
                            # down
                            cam0.continuousMove(y_speed=-0.01, stop_time=0.5)
                        elif command == '0004 0100':
                            # left
                            cam0.continuousMove(x_speed=-0.01, stop_time=0.5)
                        elif command == '0002 0100':
                            # right
                            cam0.continuousMove(x_speed=0.01, stop_time=0.5)
                        elif command == '0020 0000':
                            # zoom in
                            cam0.continuousMove(z_speed=1, stop_time=0.5)
                        elif command == '0040 0000':
                            # zoom out
                            cam0.continuousMove(z_speed=-1, stop_time=0.5)
                        else:
                            # unknown
                            print('unknown')
                            writeLog('W', 'detect', 'listenCC', 'unknown command:')


def monitorPwr(interval):
    loadStaff(staff_filename)
    while 1:
        sendPwr(loc_cn=loc_cn, device='摄像机', pwr_thr_lo=pwr_thr_lo, pwr_thr_hi=pwr_thr_hi)
        time.sleep(interval)


def putOpticalImg(optical_path, optical_queue):
    while True:
        cap = cv2.VideoCapture(optical_path)
        cap_time = datetime.datetime.now().strftime('%H.%M.%S')
        read_normal = True
        while cap.isOpened() and read_normal:
            frame = cap.read()[1]
            if frame is None:
                read_normal = False
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            optical_queue.put([frame, cap_time])
            optical_queue.get() if optical_queue.qsize() > 1 else time.sleep(0.01)



def putThermalImg(thermal_path, thermal_queue):
    while True:
        cap = cv2.VideoCapture(thermal_path)
        read_normal = True
        while cap.isOpened() and read_normal:
            frame = cap.read()[1]
            if frame is None:
                read_normal = False
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            thermal_queue.put(frame)
            thermal_queue.get() if thermal_queue.qsize() > 1 else time.sleep(0.01)


# def mainDetector(optical_path, thermal_path, mask):


def detector(optical_queue, thermal_queue, alert_queue, imgsz, interrupt, interrupted, mask=None):
    log_parameter()
    global img0_in_detect_get_time
    logger, log_init_time = init_log()
    # 写运行参数
    logger.info("")
    device = select_device('')

    model = attempt_load(weights=weight_filepath, map_location=device)  # load FP32 model
    print("Attempt load Weight from: %s" % weight_filepath)

    logger.info("Attempt load Weight from: %s" % weight_filepath)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # half = False
    if half:
        model.half()
    cudnn.benchmark = True

    names = model.module.names if hasattr(model, 'module') else model.names
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    C_PRESET_IDS = 0
    C_FOCAL_IDS = 1
    C_TIMES = 2
    C_XYXYS = 3
    C_IMGS_0 = 4
    C_IMGS_FILETIME = 5

    double_check_list = [[], [], [], [], [], []]
    smoke_fire_sent = False
    is_resume_from_double_check = False
    is_resume_from_interrupt = False
    init_optical = False
    last_send_running_time = datetime.datetime.now()
    #
    capture_path = save_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d') + '/capture'
    if not os.path.exists(capture_path):
        os.makedirs(capture_path)

    ref_img_save_path = save_path + '/' + datetime.datetime.now().strftime(
        '%Y-%m-%d') + '/debug_ref_img'
    if not os.path.exists(ref_img_save_path):
        os.makedirs(ref_img_save_path)

    send_thermal_img_path = save_path + '/' + datetime.datetime.now().strftime(
        '%Y-%m-%d') + '/fire'
    # data/Y-M-D/fire
    if not os.path.exists(send_thermal_img_path):
        os.makedirs(send_thermal_img_path)

    send_img_path = save_path + '/' + datetime.datetime.now().strftime(
        '%Y-%m-%d') + '/smoke'

    if not os.path.exists(send_img_path):
        os.makedirs(send_img_path)
    thermal_only = False
    while True:
        if datetime.datetime.now().strftime('%Y-%m-%d') != log_init_time:
            log_init_time = init_log()
        current_h_min = int(datetime.datetime.now().strftime('%H')) * 60 + int(datetime.datetime.now().strftime('%M'))

        if time_early < current_h_min < time_late:
            if thermal_only:
                thermal_only = False
                print("Optical Open")
                logger.info("Optical Open")
        else:
            if not thermal_only:
                print("Thermal Only")
                logger.info("Thermal Only")
                thermal_only = True
                init_optical = True

        if (datetime.datetime.now() - last_send_running_time).seconds > send_running_interval:
            th_sendRunning = Thread(target=sendRunning, args=[])
            th_sendRunning.start()
            last_send_running_time = datetime.datetime.now()

        if interrupt.value:
            is_resume_from_interrupt = True
            interrupted.value = 1
            print('interrupted')
            writeLog('I', 'detect', 'detector', 'interrupted by user')

            interrupt.value = 0

            auto_resume_time_old = datetime.datetime.now()

            while (datetime.datetime.now() - auto_resume_time_old).seconds < auto_resume_time:
                if interrupt.value == 1:
                    auto_resume_time_old = datetime.datetime.now()
                    interrupt.value = 0
                    interrupted.value = 1
                print('auto resume time:', auto_resume_time - (datetime.datetime.now() - auto_resume_time_old).seconds)
                time.sleep(1)

            interrupt.value = 0
            interrupted.value = 0

        if thermal_only and not interrupt.value:
            # TODO need to move to an correct angle(vertical) first.
            if not cam0.isMoving():
                cam0.continuousMove(0.01, 0, 0, 0)
                print("Thermal_only running")
            # thermal detect
            thermal_img = thermal_queue.get()
            thermal_img_time = datetime.datetime.now().strftime(
                '%H.%M.%S')
            # cv2.imshow('thermal_img', thermal_img)
            # cv2.waitKey(1)

            is_fire, thermal_detect_img = findRed(thermal_img, min_red_pixel_count)
            if is_fire:
                if not smoke_fire_sent:
                    direc = cam0.getAngle(north_angle)
                    # save_path = **/data

                    # send_thermal_img_full = send_thermal_img_path + '/' + datetime.datetime.now().strftime(
                    #     '%H.%M.%S') + '.jpg'
                    send_thermal_img_full = send_thermal_img_path + '/' + thermal_img_time + '.jpg'
                    # 图片保存路径：data/Y-M-D/fire/H-M-S.jpg
                    logger.debug("Thermal: %s.jpg---Detect fire in Thermal img" % thermal_img_time)
                    cv2.imwrite(send_thermal_img_full, thermal_detect_img)

                    alert_queue.put([cc_host, pic_port, loc_cn, direc, send_thermal_img_full, None])

                    smoke_fire_sent = True

        if not thermal_only and not interrupt.value:
            if init_optical:
                double_check_list = [[], [], [], [], [], []]
                cam0.loadPos(pos_list_file)
                init_optical = False

            # double check

            if len(double_check_list[0]) != 0:
                double_check_indexs = doubleCheckTimeOut(double_check_list[C_TIMES], double_check_interval_upper,
                                                         double_check_interval_lower)
                if len(double_check_indexs) != 0:
                    # init double check
                    if show_print_v:
                        print('***debug double check time!***', '\n', double_check_list[C_XYXYS], '\n',
                              [time_temp.strftime('%H.%M.%S') for time_temp in double_check_list[C_TIMES]], '\n')
                    print('===need double check this interval, preparing===',
                          datetime.datetime.now().strftime('%H.%M.%S'))
                    writeLog('I', 'detect', 'detector', '===need double check this interval, preparing=== ' + str(
                        datetime.datetime.now().strftime('%H.%M.%S')))

                    is_resume_from_double_check = True

                    double_check_current = []
                    for double_check_index in double_check_indexs:
                        double_check_current.append(double_check_list[C_PRESET_IDS][double_check_index])

                    print('double check list=', double_check_current, 'in all=', double_check_list[C_PRESET_IDS],
                          double_check_indexs)
                    writeLog('I', 'detect', 'detector',
                             'double check list= ' + str(double_check_current) + 'in all= ' + str(
                                 double_check_list[C_PRESET_IDS]) +
                             str(double_check_indexs))

                    for double_check_ptr, double_check_index in enumerate(double_check_indexs):
                        yolo_time = double_check_list[C_IMGS_FILETIME][double_check_index]
                        yolo_point = double_check_list[C_PRESET_IDS][double_check_index]
                        # double Check
                        # 消除回程误差
                        # if double_check_ptr == 0:  # 如果是第一个预置点
                        #     if show_print_v:
                        #         print('***debug*** extra moving for first preset')
                        #
                        #     cam0.gotoPreset(double_check_list[C_PRESET_IDS][double_check_index])
                        #     time.sleep(0.5)
                        #
                        #     removeHysterisis()
                        #
                        # else:
                        #     # print('***debug*** double_check_index=', double_check_index)
                        #     if double_check_list[C_FOCAL_IDS][double_check_indexs[double_check_ptr - 1]] != \
                        #             double_check_list[C_FOCAL_IDS][double_check_index]:
                        #         if show_print_v:
                        #             print('***debug*** extra moving for different focal',
                        #                   double_check_list[C_FOCAL_IDS][double_check_indexs[double_check_ptr - 1]],
                        #                   double_check_list[C_FOCAL_IDS][double_check_index])
                        #         cam0.gotoPreset(double_check_list[C_PRESET_IDS][double_check_index])
                        #         time.sleep(0.5)
                        #
                        #         removeHysterisis()

                        # cam0.gotoPreset(double_check_list[C_PRESET_IDS][double_check_index])
                        cam0.stable_goPreset(double_check_list[C_PRESET_IDS][double_check_index])

                        print('\ndouble check moving to', double_check_list[C_PRESET_IDS][double_check_index], 'in',
                              double_check_current)

                        writeLog('I', 'detect', 'detector', '\ndouble check moving to ' + str(
                            double_check_list[C_PRESET_IDS][double_check_index]) + ' in ' +
                                 str(double_check_current))
                        time.sleep(1)
                        is_moving = cam0.isMoving()
                        while is_moving:
                            is_moving = cam0.isMoving()

                        print('waiting for stable(double check)')
                        writeLog('I', 'detect', 'detector', 'waiting for stable(double check)')
                        # print('***debug*** before double check stable time:', datetime.datetime.now())
                        time.sleep(double_check_stable_wait_time)
                        # print('***debug*** after double check stable time:', datetime.datetime.now())

                        if show_print_v:
                            print('***debug*** current ptr:', double_check_index, 'in total:', double_check_current)
                            print('***debug*** all dc:', double_check_list[C_XYXYS])

                        double_check_times = 0
                        double_check_xyxys_tmp2 = []
                        double_check_confs_tmp2 = []
                        # 先进行一次基于Yolo的DoubleCheck 把明显不存在目标的DoubleCheck任务删除
                        # double_check_imgs = []
                        while double_check_times < double_check_detect_times:
                            # 计算得到一组 bounding box and conf

                            img, img_time = optical_queue.get()
                            # double_check_imgs.append(img)
                            img_detect2 = img.copy()
                            # img_detect2 = cv2.cvtColor(img_detect2, cv2.COLOR_BGR2GRAY)
                            # img_detect2 = cv2.cvtColor(img_detect2, cv2.COLOR_GRAY2BGR)
                            img_detect2 = letterbox(img_detect2, new_shape=imgsz)[0]
                            img_detect2 = img_detect2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                            img_detect2 = np.ascontiguousarray(img_detect2)

                            double_check_times += 1

                            with torch.no_grad():
                                xyxys2 = []
                                confs2 = []
                                img_tensor2 = torch.from_numpy(img_detect2).to(device)
                                img_tensor2 = img_tensor2.half() if half else img_tensor2.float()  # uint8 to fp16/32
                                img_tensor2 /= 255.0  # 0 - 255 to 0.0 - 1.0
                                if img_tensor2.ndimension() == 3:
                                    img_tensor2 = img_tensor2.unsqueeze(0)
                                pred2 = model(img_tensor2, augment='')[0]

                                # Apply NMS
                                pred2 = non_max_suppression(pred2, conf_thres, iou_thres)
                                for i, det in enumerate(pred2):  # detections per image
                                    s = ''
                                    if len(det):
                                        # first_check_detect_time = first_check_detect_time2

                                        for c in det[:, -1].unique():
                                            n = (det[:, -1] == c).sum()  # detections per class
                                            s += f'{n} {names[int(c)]}s, '  # add to string
                                        # Rescale boxes from img_size to im0 size
                                        det[:, :4] = scale_coords(img_tensor2.shape[2:], det[:, :4], img.shape).round()
                                        for *xyxy, conf, cls in reversed(det):
                                            if (xyxy[2] - xyxy[0]) ** 2 + (xyxy[3] - xyxy[1]) ** 2 > min_xyxy_size ** 2:
                                                xyxys2.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                                confs2.append(float(conf))

                            double_check_xyxys_tmp2.extend(xyxys2)
                            double_check_confs_tmp2.extend(confs2)
                        # 进行非极大值抑制，0.1为IOU？
                        double_check_xyxys_tmp2, double_check_confs_tmp2 = nms(double_check_xyxys_tmp2,
                                                                               double_check_confs_tmp2,
                                                                               0.1)

                        print('***debug*** before deleted', double_check_xyxys_tmp2)
                        writeLog('I', 'detect', 'detector',
                                 '***debug*** before deleted ' + str(double_check_xyxys_tmp2))

                        # print('***debug*** before deleted', double_check_list[C_XYXYS][double_check_index])

                        list_del = []
                        if len(double_check_xyxys_tmp2) == 0:
                            print('no detection in doublecheck, skip')
                            writeLog('I', 'detect', 'detector', 'no detection in doublecheck, skip')
                            logger.debug("DoubleCheck: %s_%s.jpg--Not detection in No.%d." %
                                         (yolo_time, str(yolo_point).zfill(2), double_check_times))
                        # 只有第二次Yolo也检测到才继续进行DoubleCheck
                        else:
                            # for i, xyxy_iou0 in enumerate(double_check_list[C_XYXYS][double_check_index]):
                            for i, xyxy_iou0 in enumerate(double_check_xyxys_tmp2):
                                # 计算两次检测的IOU，如果相符程度很高，则可以判读为误检
                                for xyxy_iou1 in double_check_list[C_XYXYS][double_check_index]:
                                    iou = calIou(xyxy_iou0, xyxy_iou1)
                                    print('***debug*** NO.' + str(i), iou)
                                    writeLog('I', 'detect', 'detector',
                                             '***debug*** NO. ' + str(i) + ' ' + str(iou))

                                    if iou > doucheck_iou_thd:
                                        list_del.append(i)
                                        print('***debug*** NO.' + str(i), 'in',
                                              double_check_xyxys_tmp2, 'deleted for iou>' + str(doucheck_iou_thd))
                                        writeLog('I', 'detect', 'detector',
                                                 '***debug*** NO. ' + str(i) + ' in ' +
                                                 str(double_check_xyxys_tmp2) + ' deleted for iou>0.9')
                                        logger.debug("DoubleCheck: %s_%s.jpg--IOU:%f > IOUThd:%f." %
                                                     (yolo_time, str(yolo_point).zfill(2), iou, doucheck_iou_thd,))
                            # 如果第二次YOLO没检测到，说明是误检，从DoubleCheckList里删除该项任务
                            i_del = 0
                            if len(list_del):
                                for i in list_del:
                                    del double_check_xyxys_tmp2[i - i_del]
                                    i_del += 1

                            print('***debug*** after deleted', double_check_xyxys_tmp2)
                            # print('***debug*** after deleted', double_check_list[C_XYXYS][double_check_index])
                            writeLog('I', 'detect', 'detector',
                                     '***debug*** after deleted ' + str(double_check_xyxys_tmp2))

                            if len(double_check_xyxys_tmp2) == 0:
                                print('all xyxys deleted this time, continue next')
                                writeLog('I', 'detect', 'detector', 'all xyxys deleted this time, continue next')
                                logger.debug("DoubleCheck: %s_%s.jpg--Not detect any object by Yolo in DoubleCheck"
                                             % (yolo_time, str(yolo_point).zfill(2)))
                            else:
                                # double check
                                double_check_scores = [0] * len(double_check_xyxys_tmp2)
                                # double_check_scores = [0] * len(double_check_list[C_XYXYS][double_check_index])
                                double_check_imgs1_tmp = []

                                # expend box for more detail
                                expend_ratio = 0.1

                                for i_xyxy_expend, xyxy_expend in enumerate(double_check_xyxys_tmp2):
                                    xyxy_expend0 = max(0, int(
                                        xyxy_expend[0] - (xyxy_expend[2] - xyxy_expend[0]) * expend_ratio))
                                    xyxy_expend1 = max(0, int(
                                        xyxy_expend[1] - (xyxy_expend[3] - xyxy_expend[1]) * expend_ratio))
                                    xyxy_expend2 = min(1920, int(
                                        xyxy_expend[2] + (xyxy_expend[2] - xyxy_expend[0]) * expend_ratio))
                                    xyxy_expend3 = min(1080, int(
                                        xyxy_expend[3] + (xyxy_expend[3] - xyxy_expend[1]) * expend_ratio))
                                    double_check_xyxys_tmp2[i_xyxy_expend][0] = xyxy_expend0
                                    double_check_xyxys_tmp2[i_xyxy_expend][1] = xyxy_expend1
                                    double_check_xyxys_tmp2[i_xyxy_expend][2] = xyxy_expend2
                                    double_check_xyxys_tmp2[i_xyxy_expend][3] = xyxy_expend3

                                # find maxmalRectangle
                                xyxys_half = [[int(i / 2) for i in xyxy_] for xyxy_ in
                                              doublene_check_xyxys_tmp2]  # boxes shrink to 0.5x boxes
                                rect_half = maxmalRectangle((int(1080 / 2), int(1920 / 2)),
                                                            xyxys_half)  # calculate max rect in pic
                                newboxes_half = findboxes4boxes(xyxys_half, rect_half,
                                                                double_check_cal_times)  # calculate ref boxes in max rect
                                ref_xyxys = [[[i * 2 for i in newboxi_half] for newboxi_half in newbox_half]
                                             for newbox_half in newboxes_half]  # boxes expand 2x boxes

                                double_check_times = 0
                                # one object run doubleCheck * times
                                double_check_dist_list = []
                                double_check_score_list = []
                                # 根据距离计算进行DoubleCheck
                                logger.debug(
                                    "DoubleCheck: %s_%s.jpg--BoundingBox: %s -- RefBox: %s." %
                                    (yolo_time, str(yolo_point).zfill(2), str(double_check_xyxys_tmp2), str(ref_xyxys)))

                                while double_check_times < double_check_cal_times:
                                    print('double checking...', double_check_list[C_PRESET_IDS][double_check_index],
                                          end='\r')
                                    writeLog('I', 'detect', 'detector', 'double checking... ' + str(
                                        double_check_list[C_PRESET_IDS][double_check_index]))

                                    img, img_time = optical_queue.get()
                                    double_check_imgs1_tmp.append(img)

                                    ref_img_save_path_full_i = \
                                        ref_img_save_path + '/' + yolo_time + '_' \
                                        + str(double_check_list[C_PRESET_IDS][double_check_index]).zfill(2) \
                                        + '_' + str(double_check_times + 1) + '.jpg'

                                    cv2.imwrite(ref_img_save_path_full_i,
                                                img)

                                    # debug
                                    if show_img:
                                        cv2.namedWindow('first', cv2.WINDOW_FREERATIO)
                                        cv2.imshow('first', img_plot)
                                        cv2.waitKey(1)
                                    # cal (img0, img1)'s score
                                    double_check_score_delta, dist_list = doubleCheck(
                                        img0=double_check_list[C_IMGS_0][double_check_index][-1],
                                        img1=img, xyxys=double_check_xyxys_tmp2,
                                        ref_xyxys=ref_xyxys[double_check_times],
                                        score_add=double_check_score_add,
                                        score_minus=double_check_score_minus,
                                        thd=double_check_dist_thr)
                                    # 得到一次距离[[yolo1,yolo2,...]]
                                    double_check_dist_list.append(dist_list)
                                    double_check_score_list.append(double_check_scores)
                                    for i_score in range(len(double_check_scores)):
                                        double_check_scores[i_score] += double_check_score_delta[i_score]

                                    print('delta', double_check_score_delta, 'score', double_check_scores)
                                    double_check_times += 1

                                # compare doubleCheckScore got by DoubleCheck accumulation with thd
                                # 如果小于设定阈值 --> 直接判定无烟雾
                                double_check_dist_list_np = np.asarray(double_check_dist_list)
                                double_check_dist_list_ = []
                                # transform list
                                for i in range(len(double_check_dist_list_np[0])):
                                    double_check_dist_list_.append(double_check_dist_list_np[:, i].tolist())

                                if max(double_check_scores) < double_check_score_thd:
                                    # All bounding box no smoke
                                    print('not smoke')
                                    writeLog('I', 'detect', 'detector', 'not smoke')
                                    logger.debug(
                                        "DoubleCheck: %s_%s.jpg--Not smoke. "
                                        "Dist: %s, Compared with :%s. Scores: %s Compared with Thd %s"
                                        % (yolo_time, str(yolo_point).zfill(2), str(double_check_dist_list_),
                                           str(double_check_dist_thr), str(double_check_scores),
                                           str(double_check_score_thd)))
                                else:
                                    print('is smoke')
                                    writeLog('I', 'detect', 'detector', 'is smoke')

                                    for i, score in enumerate(double_check_scores):
                                        if score >= double_check_score_thd:
                                            plot_one_box(double_check_xyxys_tmp2[i],
                                                         img,
                                                         color=(0, 0, 255),
                                                         line_thickness=5)

                                    # save img
                                    send_img_full = send_img_path + '/' + yolo_time + '_' + str(
                                        double_check_list[C_PRESET_IDS][double_check_index]) + '.jpg'

                                    send_video_path_full = send_img_path + '/' + yolo_time + '_' + str(
                                        double_check_list[C_PRESET_IDS][double_check_index]).zfill(2) + '.mp4'

                                    cv2.imwrite(send_img_full, img)

                                    video_send = double_check_list[C_IMGS_0][double_check_index]
                                    video_send.extend(double_check_imgs1_tmp)
                                    video_send.append(img)
                                    imgs2video(video_send, send_video_path_full, 3, 1920, 1080)

                                    direc = cam0.getAngle(north_angle)
                                    print('alert sent', loc_cn, direc)
                                    writeLog('I', 'detect', 'detector', 'alert sent')

                                    alert_queue.put(
                                        [cc_host, pic_port, loc_cn, direc, send_img_full, send_video_path_full])
                                    logger.debug(
                                        "DoubleCheck: %s_%s.jpg--Smoke Confirmed and Send Alert. "
                                        "Dist: %s, Compared with :%s. Scores: %s Compared with Thd %s"
                                        % (yolo_time, str(yolo_point).zfill(2), str(double_check_dist_list_),
                                           str(double_check_dist_thr), str(double_check_scores),
                                           str(double_check_score_thd)))

                    if show_print_v:
                        print('debug length before= ',
                              len(double_check_list[C_PRESET_IDS]),
                              len(double_check_list[C_FOCAL_IDS]),
                              len(double_check_list[C_TIMES]),
                              len(double_check_list[C_XYXYS]),
                              len(double_check_list[C_IMGS_0]),
                              len(double_check_list[C_IMGS_FILETIME])
                              )

                    i_del = 0
                    for double_check_index in double_check_indexs:
                        del double_check_list[C_PRESET_IDS][double_check_index - i_del]
                        del double_check_list[C_FOCAL_IDS][double_check_index - i_del]
                        del double_check_list[C_TIMES][double_check_index - i_del]
                        del double_check_list[C_XYXYS][double_check_index - i_del]
                        del double_check_list[C_IMGS_0][double_check_index - i_del]
                        del double_check_list[C_IMGS_FILETIME][double_check_index - i_del]
                        i_del += 1

                    if show_print_v:
                        print('debug length= ',
                              len(double_check_list[C_PRESET_IDS]),
                              len(double_check_list[C_FOCAL_IDS]),
                              len(double_check_list[C_TIMES]),
                              len(double_check_list[C_XYXYS]),
                              len(double_check_list[C_IMGS_0]),
                              len(double_check_list[C_IMGS_FILETIME]),
                              )

                    print('double check finished this interval')
                    writeLog('I', 'detect', 'detector', 'double check finished this interval')

            # first check

            current_id, is_first_preset, focal_id = cam0.nextPos()
            print('\nmoving to', current_id)
            writeLog('I', 'detect', 'detector', 'moving to ' + str(current_id))

            time.sleep(0.5)
            is_moving = cam0.isMoving()
            while is_moving:
                is_moving = cam0.isMoving()

            if is_first_preset or is_resume_from_double_check or is_resume_from_interrupt:
                # 如果被打断了，需要重新回到firstcheck，也需要消除回程误差
                is_resume_from_double_check = False
                is_resume_from_interrupt = False

                # removeHysterisis()
                #
                # cam0.gotoPreset(current_id)
                cam0.stable_goPreset(current_id)

                print('waiting for stable(first check extra)')
                time.sleep(first_check_stable_wait_time_focal)
                writeLog('I', 'detect', 'detector', 'waiting for stable(first check extra)')

            else:
                print('waiting for stable(first check)')
                time.sleep(first_check_stable_wait_time)
                writeLog('I', 'detect', 'detector', 'waiting for stable(first check)')

            current_time = datetime.datetime.now()
            print('detecting...')
            writeLog('I', 'detect', 'detector', 'detecting...')

            started_detect_time = current_time

            smoke_fire_sent = False
            double_check_imgs0_tmp = []
            double_check_xyxys_tmp = []
            double_check_confs_tmp = []
            first_check_detect_time = first_check_detect_time_long
            # 在一个预置点检测数次，
            while (current_time - started_detect_time).seconds < first_check_detect_time:
                if (current_time - started_detect_time).seconds >= first_check_detect_time_short and len(
                        double_check_xyxys_tmp) == 0:
                    print('no detection for 1 second, exit')
                    writeLog('I', 'detect', 'detector', 'no detection for 1 second, exit')

                    double_check_imgs0_tmp = []
                    double_check_xyxys_tmp = []
                    double_check_confs_tmp = []
                    break

                # thermal detect
                thermal_img = thermal_queue.get()
                is_fire, thermal_detect_img = findRed(thermal_img, min_red_pixel_count)
                if is_fire:
                    if not smoke_fire_sent:
                        direc = cam0.getAngle(north_angle)
                        send_thermal_img_path = save_path + '/' + datetime.datetime.now().strftime(
                            '%Y-%m-%d') + '/fire'
                        if not os.path.exists(send_thermal_img_path):
                            os.makedirs(send_thermal_img_path)
                        send_thermal_img_full = send_thermal_img_path + '/' + datetime.datetime.now().strftime(
                            '%H.%M.%S') + '.jpg'
                        cv2.imwrite(send_thermal_img_full, thermal_detect_img)
                        alert_queue.put([cc_host, pic_port, loc_cn, direc, send_thermal_img_full, None])

                        smoke_fire_sent = True

                img, img0_in_detect_get_time = optical_queue.get()


                img_detect = img.copy()
                # img_detect = cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY)
                # img_detect = cv2.cvtColor(img_detect, cv2.COLOR_GRAY2BGR)
                img_detect = letterbox(img_detect, new_shape=imgsz)[0]
                img_detect = img_detect[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img_detect = np.ascontiguousarray(img_detect)

                current_time = datetime.datetime.now()

                with torch.no_grad():
                    xyxys = []
                    confs = []
                    img_plot = img.copy()
                    img_tensor = torch.from_numpy(img_detect).to(device)
                    img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
                    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img_tensor.ndimension() == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    pred = model(img_tensor, augment='')[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres)
                    for i, det in enumerate(pred):  # detections per image
                        s = ''
                        if len(det):
                            # first_check_detect_time = first_check_detect_time2

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f'{n} {names[int(c)]}s, '  # add to string
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
                            for *xyxy, conf, cls in reversed(det):
                                if (xyxy[2] - xyxy[0]) ** 2 + (xyxy[3] - xyxy[1]) ** 2 > min_xyxy_size ** 2:
                                    xyxys.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                                    confs.append(float(conf))
                                    plot_one_box(xyxy, img_plot, label='smoke', color=[255, 0, 0],
                                                 line_thickness=2)
                    # debug
                    if show_img:
                        cv2.namedWindow('first', cv2.WINDOW_FREERATIO)
                        cv2.imshow('first', img_plot)
                        cv2.waitKey(1)

                double_check_imgs0_tmp.append(img)
                double_check_xyxys_tmp.extend(xyxys)
                double_check_confs_tmp.extend(confs)

            # if got detection, append ref images and ids
            if len(double_check_xyxys_tmp):
                double_check_xyxys_tmp, double_check_confs_tmp = removeFew(double_check_xyxys_tmp,
                                                                           double_check_confs_tmp,
                                                                           few_remove_count_thr, few_remove_iou_thr)
                double_check_xyxys_tmp, double_check_confs_tmp = nms(double_check_xyxys_tmp, double_check_confs_tmp,
                                                                     0.1)

                if len(double_check_xyxys_tmp) == 0:
                    print('too few, ignored')
                    writeLog('I', 'detect', 'detector', 'too few, ignored')

                else:
                    print('Preset detection: recorded:', current_id)
                    writeLog('I', 'detect', 'detector', 'preset detection: recorded: ' + str(current_id))

                    # print(double_check_list)
                    double_check_list[C_PRESET_IDS].append(current_id)
                    double_check_list[C_FOCAL_IDS].append(focal_id)
                    double_check_list[C_TIMES].append(datetime.datetime.now())
                    double_check_list[C_XYXYS].append(double_check_xyxys_tmp)
                    double_check_list[C_IMGS_0].append(double_check_imgs0_tmp)
                    double_check_list[C_IMGS_FILETIME].append(img0_in_detect_get_time)

                    ref_img_save_path_full_0 = \
                        ref_img_save_path + '/' + img0_in_detect_get_time + '_' \
                        + str(current_id).zfill(2) + '_0.jpg'

                    cv2.imwrite(ref_img_save_path_full_0,
                                img)
                    logger.debug(
                        "YOLO: Detected smoke at Point:%d. File: %s.jpg" % (current_id, img0_in_detect_get_time))
            # Capture when not first run and not moving

            cv2.imwrite(capture_path + '/' + img0_in_detect_get_time + '_' + str(
                current_id) + '.jpg', img)


"""==== load config ===="""
config = configparser.ConfigParser()
root_dir = os.getcwd()
config.read(root_dir + '/data/config.ini', encoding='utf-8')

# camera
onvif_ip = config.get('camera', 'ip')
onvif_user = config.get('camera', 'user')
onvif_pwd = config.get('camera', 'password')
cc_host = config.get('camera', 'control_center_host')
cc_port = config.getint('camera', 'control_center_port')
pic_port = config.getint('camera', 'alert_image_port')
dev_id = config.getint('camera', 'dev_id')

north_angle = config.getfloat('camera', 'north_angle')
optical_path = 'rtsp://admin:a1234567@192.168.1.65:554/Streaming/Channels/1'
thermal_path = 'rtsp://admin:a1234567@192.168.1.65:554/Streaming/Channels/201'

# main
loc_cn = config.get('main', 'loc_cn')

# files
files_dir = os.path.join(os.getcwd(), 'data')
pos_list_file = config.get('files', 'position_list')
pos_list_file = pos_list_file.replace(' ', '').split(',')
pos_list_file = [os.path.join(files_dir, filename) for filename in pos_list_file]

staff_list_file = os.path.join(files_dir, config.get('files', 'staff_list'))

staff_filename = os.path.join(files_dir, config.get('files', 'staff_list'))
print('Load staff file from:%s' % staff_filename)

weights_dir = config.get('files', 'weight_dir')
weight_filepath = os.path.join(weights_dir, config.get('files', 'weight'))

# path
save_path = config.get('path', 'save_path')

mask_filepath = root_dir + '/data/mask/'

# para
doucheck_iou_thd = config.getfloat('parameter', 'doucheck_iou_thd')
double_check_dist_thr = config.getint('parameter', 'double_check_dist_thr')
# 2000 for extra high, 2800 for high, 5000 for medium, 8000 for low, 10000 for extra low

"""========= init parameter ========="""

# init camera
cam0 = OnvifHik(onvif_ip, '80', onvif_user, onvif_pwd)
cam0.contactCam()
cam0.loadPos(pos_list_file)
loadStaff(staff_list_file)

# Parameter setting
first_check_stable_wait_time = 1
first_check_stable_wait_time_focal = 3

first_check_detect_time_short = 1
first_check_detect_time_long = 2.5

min_red_pixel_count = 5

double_check_stable_wait_time = 3.5
double_check_detect_times = 5
double_check_cal_times = 7
double_check_interval_upper = 90
double_check_interval_lower = 60

double_check_score_add = 1
double_check_score_minus = 1
double_check_score_thd = 5

# few remove
few_remove_count_thr = 4
few_remove_count_thr2 = 2
few_remove_iou_thr = 0.1

# power thresholds
pwr_thr_lo = 23
pwr_thr_hi = 24

time_early_H = 6
time_early_M = 50
time_late_H = 20
time_late_M = 00

time_early = time_early_H * 60 + time_early_M
time_late = time_late_H * 60 + time_late_M

imgsz = 640
conf_thres = 0.2
iou_thres = 0.2

min_xyxy_size = 60

show_img = True
show_print = True
show_print_v = False

auto_resume_time = 60
send_running_interval = 600

ContorlCenter_f = 0

if __name__ == '__main__':
    # Init sending volt thread
    thread_pwr = Thread(target=monitorPwr, args=[600], daemon=True)
    thread_pwr.start()

    mp.set_start_method('spawn')  # init

    interrupt = mp.Value('i', 0)
    interrupted = mp.Value('i', 0)
    optical_queue = mp.Queue(maxsize=2)
    time_queue = mp.Queue(maxsize=2)
    thermal_queue = mp.Queue(maxsize=2)
    alert_queue = mp.Queue()
    processes = [mp.Process(target=putOpticalImg, args=(optical_path, optical_queue, time_queue)),
                 mp.Process(target=putThermalImg, args=(thermal_path, thermal_queue)),
                 mp.Process(target=detector,
                            args=(
                                optical_queue, time_queue,thermal_queue, alert_queue, 640, interrupt, interrupted)),
                 mp.Process(target=listenControlCenter, args=(cc_host, cc_port, dev_id, interrupt, interrupted)),
                 mp.Process(target=alert, args=(alert_queue,))]

    [process.start() for process in processes]
    [process.join() for process in processes]
