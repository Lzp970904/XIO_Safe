# ----------------------- 需要检测的视频流相关配置 ----------------------

# 各视频流是否检测的开关，True 表示检测， False 表示不检测
switch_dict = {
    'sawanini_1': True,
    'sawanini_2': True,
    'zhuanjixia': True,
    'penfenshang': True,
    'baobantongyong': True,
}
# 是否连接OPC服务器，执行紧急停机
open_opc = False
# 开启邮箱OPC报警
open_email_warning = False
# 开启统计闯入次数和邮箱发送报告功能
open_email_report = True
# 开启数据库存储异常记录
open_mysql_save_record = True

# 视频流 URL
# video_stream_paths_dict = {
#     'sawanini_1': 'rtsp://user:xiolift123@10.19.31.138/Streaming/Channels/102',
#     'sawanini_2': 'rtsp://user:xiolift123@10.19.31.154/Streaming/Channels/102',
#     'zhuanjixia': 'rtsp://user:xiolift123@10.19.31.136/Streaming/Channels/102',
#     'penfenshang': 'rtsp://user:xiolift123@10.19.31.139/Streaming/Channels/102',
#     'baobantongyong': 'rtsp://user:xiolift123@10.19.31.137/Streaming/Channels/102',
# }

# for local test
video_stream_paths_dict = {
    'sawanini_1': 'videos/01.avi',
    'sawanini_2': 'videos/02.avi',
    'zhuanjixia': 'videos/03.avi',
    'penfenshang': 'videos/04.avi',
    'baobantongyong': 'videos/05.avi',
}

# video_stream_paths_dict = {
#     'sawanini_1': 'rtsp://admin:hdu417417@192.168.2.2/Streaming/Channels/102',
#     'sawanini_2': 'rtsp://admin:hdu417417@192.168.2.3/Streaming/Channels/102',
#     'zhuanjixia': 'rtsp://admin:hdu417417@192.168.2.4/Streaming/Channels/102',
#     'penfenshang': 'rtsp://admin:hdu417417@192.168.2.8/Streaming/Channels/102',
#     'baobantongyong': 'rtsp://admin:hdu417417@192.168.2.5/Streaming/Channels/102',
# }

# 对应工位名称
stations_name_dict = {
    'sawanini_1': '萨瓦尼尼-1',
    'sawanini_2': '萨瓦尼尼-2',
    'zhuanjixia': '专机下线',
    'penfenshang': '喷粉上线',
    'baobantongyong': '薄板通用线',
}

# 视频帧的形状
frame_shape = (480, 640)

# --------------------------------------------------------------------


# ----------------------- 异常闯入报告功能相关配置 ----------------------
# 是否开启统计闯入次数和邮箱发送报告功能
# open_email_report = False
# 邮箱发送统计报告的时间间隔(s)
report_statistics_interval = 3600

# 是否开启微信机器人
open_wechat_bot = False
# 发送的群组 群组必须被添加到通讯录
wechat_group = "机器人安全监测"
# 发送异常信息的时间间隔(秒)
wechat_send_interval = 30

# 存储异常记录的数据库相关配置
# open_mysql_save_record = False
mysql_host = 'localhost'
mysql_user = 'root'
mysql_password = '123456'
mysql_db = 'python_test'
# --------------------------------------------------------------------

# 主显示的视频流名称变量
vis_name = 'sawanini_2'
prevs_vis_name = vis_name


# ------------------- 判断检测框是否闯入禁区相关配置 ---------------------
# 判断人的边界框进入禁区的阈值
inter_threshold = 0.15

# 禁区掩码图像路径
masks_paths_dict = {
    'sawanini_1': 'images/masks/mask01.jpg',
    'sawanini_2': 'images/masks/mask02.jpg',
    'zhuanjixia': 'images/masks/mask03.jpg',
    'penfenshang': 'images/masks/mask04.jpg',
    'baobantongyong': 'images/masks/mask05.jpg',
}

# 设定待检测目标物体最大边界框面积，
# 排除模型误检中较大的物体
max_object_bbox_area_dict = {
    'sawanini_1': 15000,
    'sawanini_2': 12000,
    'zhuanjixia': 12000,
    'penfenshang': 13000,
    'baobantongyong': 20000,  # TODO
}
# 设定待检测目标物体最小边界框面积，
# 排除模型误检中较小的物体
min_object_bbox_area_dict = {
    'sawanini_1': 500,
    'sawanini_2': 500,
    'zhuanjixia': 500,
    'penfenshang': 100,
    'baobantongyong': 100,
}

# 排除掉可能被错识别为人的目标物体区域
# (x1, y1, x2, y2)
excluded_objects_dict = {
    'sawanini_1': [],
    'sawanini_2': [],
    'zhuanjixia': [(175, 130, 200, 155), ],
    'penfenshang': [(270, 180, 350, 270), ],  # (310, 230)
    'baobantongyong': [],
}
# ----------------------------------------------------------------------


# ----------------------- OPC服务相关配置 -------------------------------
# 是否连接OPC服务器，执行紧急停机
# open_opc = False

# OPC 服务器 URL
opc_url = 'opc.tcp://10.19.3.49:49320'  # 本机OPC服务

# 机器工位节点
nodes_dict = {
    'sawanini_1': 'ns=1;s=sawaninigaopin.OP30-S71214C.安全检测触发',
    'sawanini_2': 'ns=1;s=xinsawaninihoudaoxianti.QCPU.光栅触发暂停标志',
    'zhuanjixia': 'ns=1;s=laozhuanjixian.shangpenfenxiazhewanCJ2M.下折弯安全检测触发',
    'penfenshang': 'ns=1;s=laozhuanjixian.shangpenfenxiazhewanCJ2M.上喷粉安全检测触发',
    'baobantongyong': 'ns=1;s=weibentongyongxian.S7300.安全检测触发',
}

nodes_dict_test = {
    'sawanini_1': 'ns=1;s=sawaninigaopin.OP30-S71214C',
    'sawanini_2': 'ns=1;s=xinsawaninihoudaoxianti.QCPU',
    'zhuanjixia': 'ns=1;s=laozhuanjixian.shangpenfenxiazhewanCJ2M',
    'penfenshang': 'ns=1;s=laozhuanjixian.shangpenfenxiazhewanCJ2M',
    'baobantongyong': 'ns=1;s=weibentongyongxian.S7300',
}

# 轮巡OPC节点状态的间隔时间(s)
patrol_opc_nodes_interval = 30
# 是否开启邮箱OPC报警
# open_email_warning = False
# 邮箱报警间隔时间(s)
email_opc_warning_interval = 3600
# 是否开启通过弹窗报警
open_popup_message_box = False
# ---------------------------------------------------------------------------

# ----------------------- 系统保卫进程相关配置 -------------------------------
# 检测的主循环更新 detection_flag 的时间间隔(s)
update_detection_flag_interval = 20

# 保卫进程读取 detection_flag 值的时间间隔(s) 应大于update_detection_flag_interval
check_detection_process_interval = 65
# ---------------------------------------------------------------------------

# ------------------- detection model configuration -------------------------
img_size = 416  # size of each image dimension
config_path = 'configs/yolov3.cfg'  # path to model configs file
weights_path = 'weights/yolov3.weights'  # path to weights file
class_path = 'configs/coco.names'  # path to class label file
conf_thres = 0.8  # object confidence threshold
nms_thres = 0.4  # iou threshold for non-maximum suppression
device_name = 'cuda:0'  # use cuda if available
# ---------------------------------------------------------------------------
