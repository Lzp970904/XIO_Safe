import threading
import time
import cv2
import copy
import socket
import sys
import numpy as np
import json
from queue import LifoQueue


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


# 建立sock连接
# address要连接的服务器IP地址和端口号
def camera1():
    address = ('10.1.18.232', 9000)
    try:
        # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    # # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
    # ret, frame = capture.read()
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        ret, frame = cap.read()
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        # 建立矩阵
        data = np.array(imgencode)
        # 将numpy矩阵转换成字符形式，以便在网络中传输
        stringData = data.tostring()

        # 先发送要发送的数据的长度
        # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        sock.send(str.encode(str(len(stringData)).ljust(16)))
        # 发送数据
        sock.send(stringData)
        # 读取服务器返回值
        print("1")
        data = sock.recv(10240)
        print(data)
        if len(data) == 0:
            cv2.imshow("a", frame)
            cv2.waitKey(1)
            continue
        mylist = json.loads(data)
        # if mylist is None:
        #     cv2.imshow("a", frame)
        #     cv2.waitKey(1)
        #     continue
        # if len(mylist) == 0:
        #     cv2.imshow("a", frame)
        #     cv2.waitKey(1)
        #     continue
        for rect1 in mylist:
            text = rect1[1][0]
            cv2.putText(frame, text, (int(rect1[0][0][0]), int(rect1[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.line(frame, (int(rect1[0][0][0]), int(rect1[0][0][1])),
                     (int(rect1[0][1][0]), int(rect1[0][1][1])),
                     (0, 0, 255), 2)
            cv2.line(frame, (int(rect1[0][1][0]), int(rect1[0][1][1])),
                     (int(rect1[0][2][0]), int(rect1[0][2][1])),
                     (0, 0, 255), 2)
            cv2.line(frame, (int(rect1[0][2][0]), int(rect1[0][2][1])),
                     (int(rect1[0][3][0]), int(rect1[0][3][1])),
                     (0, 0, 255), 2)
            cv2.line(frame, (int(rect1[0][3][0]), int(rect1[0][3][1])),
                     (int(rect1[0][0][0]), int(rect1[0][0][1])),
                     (0, 0, 255), 2)
        # q1.put(frame)
        cv2.imshow("lala", frame)
        cv2.waitKey(1)


camera1()
