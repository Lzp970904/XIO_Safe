import time
import logging

import cv2
import torch
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QSize

from model.models import Darknet
from video_stream.visualize import Visualize
from handler.opc_client import OpcClient
from configs.config import *
from utils.utils import non_max_suppression, load_classes, calc_fps
from model.transform import transform, stack_tensors, preds_postprocess
from handler.intrusion_handling import IntrusionHandling
from handler.send_email import Email
from video_stream.video_stream import VideoLoader
from configs.config import patrol_opc_nodes_interval, update_detection_flag_interval,\
    open_email_warning, stations_name_dict
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_model(config_path, img_size, weights_path, device):
    model = Darknet(config_path, img_size=img_size)
    model.load_darknet_weights(weights_path)

    model = model.to(device)
    model.eval()  # Set in evaluation mode
    return model


# model: YOLO模型
# input_tensor: 一个Tensor
# device: cuda0
# num_classes: ？
# conf_thres: 置信度阈值
# nms_thres: ？
def inference(model, input_tensor, device, num_classes, conf_thres, nms_thres):
    try:
        torch.cuda.empty_cache()  # 修复 RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
        input_tensor = input_tensor.to(device)
        # print(input_tensor.shape)

        output = model(input_tensor)
        preds = non_max_suppression(output, conf_thres, nms_thres)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        preds = [None for _ in range(input_tensor.shape[0])]
        print(e)
        logging.error(e)
    # preds = non_max_suppression(output, num_classes, conf_thres, nms_thres)
    return preds


def array_to_QImage(img, size):
    rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytes_per_line = ch * w
    qimage = QImage(rgbImage.data, w, h, bytes_per_line, QImage.Format_RGB888)
    if isinstance(size, QSize):
        qimage = qimage.scaled(size)
    else:
        qimage = qimage.scaled(size[0], size[1])
    return qimage


def change_vis_stream(index):
    global vis_name
    global prevs_vis_name

    prevs_vis_name = vis_name
    vis_name = list(video_stream_paths_dict.keys())[index]


@torch.no_grad()
def detect_main(q_thread):
    q_thread.status_update.emit('模型加载')
    device = torch.device(device_name)
    model = get_model(config_path, img_size, weights_path, device)
    logging.info('Model initialized')

    q_thread.status_update.emit('初始化异常处理程序')
    visualize = Visualize(masks_paths_dict)
    opc_client = None
    handling = IntrusionHandling(masks_paths_dict, opc_client)

    q_thread.status_update.emit('读取视频流')
    video_loader = VideoLoader(video_stream_paths_dict)

    q_thread.status_update.emit('准备就绪')
    logging.info('Video stream create: ' + ', '.join(n for n in video_stream_paths_dict.keys()))
    classes = load_classes(class_path)

    # 计时器开始计时
    since = update_detection_flag_clock_start = time.time()

    # 用于后续计算检测FPS
    accum_time, curr_fps = 0, 0
    show_fps = 'FPS: ??'
    while True:
        crr_time = time.time()

        if crr_time - update_detection_flag_clock_start > update_detection_flag_interval:
            update_detection_flag_clock_start = crr_time
            q_thread.detection_flag.value = 1

        vis_imgs_dict = video_loader.getitem()

        active_streams = []
        input_tensor = []
        for name in vis_imgs_dict.keys():
            if vis_imgs_dict[name] is None:
                if prevs_frames_dict is not None:
                    vis_imgs_dict[name] = prevs_frames_dict[name]
                    # 将图片转换成 PyTorch Tensor
                    tensor = transform(vis_imgs_dict[name], img_size)
                    input_tensor.append(tensor)
            else:
                active_streams.append(stations_name_dict[name])
                tensor = transform(vis_imgs_dict[name], img_size)
                input_tensor.append(tensor)

        if len(input_tensor) == len(vis_imgs_dict):
            prevs_frames_dict = vis_imgs_dict
        elif len(input_tensor) == 0:
            print("未读到任何视频帧")
            time.sleep(0.5)
            continue
        # 将多张图片的Tensor堆叠一起，相当于batch size
        input_tensor = stack_tensors(input_tensor)

        # model inference and postprocess
        preds = inference(model, input_tensor, device, 80, conf_thres, nms_thres)

        if prevs_frames_dict is None:
            not_none_streams = [x for x in vis_imgs_dict.keys() if vis_imgs_dict[x] is not None]
        else:
            not_none_streams = list(vis_imgs_dict.keys())
        # 返回值只有非None视频流的预测结果
        preds_dict = preds_postprocess(preds, not_none_streams, frame_shape, img_size, classes)

        judgements_dict = handling.judge_intrusion(preds_dict)

        # calculate inference fps
        since, accum_time, curr_fps, show_fps = calc_fps(since, accum_time, curr_fps, show_fps)

        vis_imgs_dict = visualize.draw(vis_imgs_dict, preds_dict, judgements_dict, show_fps)

        handling.handle_judgement(judgements_dict, vis_imgs_dict)

        # emit the information to the front end
        if vis_name in vis_imgs_dict:
            img = vis_imgs_dict[vis_name]
            qsize = q_thread.main_window.videoLabel_1.size()
            qimage = array_to_QImage(img, qsize)
            q_thread.video_1_change_pixmap.emit(qimage)

        for name in judgements_dict.keys():
            if judgements_dict[name]:
                timestr = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())
                q_thread.text_append.emit(timestr + name + ' 启动联锁保护')
                # show intrusion record
                img = vis_imgs_dict[name]
                qsize = q_thread.main_window.recordLabel.size()
                qimage = array_to_QImage(img, qsize)
                q_thread.record_change_pixmap.emit(qimage)

        if prevs_vis_name in vis_imgs_dict:
            prevs_img = vis_imgs_dict[prevs_vis_name]
            vis_imgs_dict[vis_name] = prevs_img
            vis_imgs_dict.pop(prevs_vis_name)

        for i, img in enumerate(vis_imgs_dict.values()):
            qsize = q_thread.main_window.videoLabel_2.size()
            qimage = array_to_QImage(img, qsize)
            if i == 0:
                q_thread.video_2_change_pixmap.emit(qimage)
            elif i == 1:
                q_thread.video_3_change_pixmap.emit(qimage)
            elif i == 2:
                q_thread.video_4_change_pixmap.emit(qimage)
            elif i == 3:
                q_thread.video_5_change_pixmap.emit(qimage)
            else:
                raise RuntimeError("No so many QLabel!")
