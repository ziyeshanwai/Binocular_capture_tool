import queue
import threading
import time
from socket import *
import numpy as np
import cv2


class SocketCameraThread(threading.Thread):
    def __init__(self, thread_id, target_ip, target_port):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.img_bytes_queue = queue.Queue()
        self.target_ip = target_ip  # ip
        self.target_port = target_port  # port
        self.tcpCliSock = socket(AF_INET, SOCK_STREAM)

    def init_connection(self):
        self.connection()
        self.send_message(self.threadID)

    def run(self):
        self.init_connection()
        print("线程 %d 开启" % self.threadID)
        # TODO
        self.receive_message()
        print("线程 %d 结束" % self.threadID)

    def connection(self):
        """
        build connection with the target ip and port
        :return:
        """
        address = (self.target_ip, self.target_port)
        self.tcpCliSock.connect(address)

    def send_message(self, str_message):
        """
        send message
        :return:
        """
        self.tcpCliSock.send(bytes(str_message, 'utf-8'))  # 客户端发送消息，必须发送字节数组

    def receive_message(self):
        """
        receive the message from the server
        :return:
        """

        while True:
            buffer = self.tcpCliSock.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组
            if b'\r\n\r\n' in buffer:
                if len(buffer.split(b"\r\n\r\n")) > 2:
                    continue
                else:
                    start = time.time()
                    header, img_part = buffer.split(b"\r\n\r\n")
                    img_length = int(header.split(b"Content-length: ")[1])
                    while len(img_part) != img_length:
                        buffer = self.tcpCliSock.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组12321
                        img_part += buffer
                    if len(img_part == img_length):
                        """
                        添加到队列
                        """
                        self.img_bytes_queue.put(img_part)  ## only add to the queue


def decode_img(img_bytes):
    """
    decode the bytes img to numpy format
    :param img_bytes:
    :return: numpy format img
    """
    nparr = np.fromstring(img_bytes, np.uint8)
    img_decode = cv2.imdecode(nparr, -1)
    return img_decode


if __name__ == '__main__':
    thread_id_1 = "left"
    thread_id_2 = "right"
    target_ip = "192.168.106.80"
    target_port_1 = "8080"
    target_port_2 = "8081"
    left_camera_client = SocketCameraThread(thread_id_1, target_ip, target_port_1)
    right_camera_client = SocketCameraThread(thread_id_2, target_ip, target_port_2)
    left_camera_client.start()
    right_camera_client.start()
    while True:
        if not left_camera_client.img_bytes_queue.empty() and not right_camera_client.img_bytes_queue.empty():
            left_img_bytes = left_camera_client.img_bytes_queue.get()
            right_img_bytes = right_camera_client.img_bytes_queue.get()
            left_img = decode_img(left_img_bytes)
            right_img = decode_img(right_img_bytes)
            cv2.imshow("left", left_img)
            cv2.imshow("right", right_img)
            cv2.waitKey(1)
        else:
            time.sleep(1)

