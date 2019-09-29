#coding:utf-8

from socket import *
import cv2
import numpy as np
import time


if __name__ == "__main__":

    # print("=====================SocketServer TCP客户端=====================")
    HOST = '192.168.106.80'  #本机测试
    # HOST = '127.0.0.1'  # 本机测试
    PORT_1 = 8080
    PORT_2 = 8081
    BUFSIZ = 4096*5
    ADDR_1 = (HOST, PORT_1)
    ADDR_2 = (HOST, PORT_2)
    tcpCliSock_1 = socket(AF_INET, SOCK_STREAM)  # 创建客户端套接字
    tcpCliSock_1.connect(ADDR_1)  # 发起TCP连接
    tcpCliSock_2 = socket(AF_INET, SOCK_STREAM)  # 创建客户端套接字
    tcpCliSock_2.connect(ADDR_2)  # 发起TCP连接
    tcpCliSock_1.send(bytes("start_1", 'utf-8'))  # 客户端发送消息，必须发送字节数组
    tcpCliSock_2.send(bytes("start_2", 'utf-8'))  # 客户端发送消息，必须发送字节数组

    while True:
        buffer_1 = tcpCliSock_1.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组12321
        if b'\r\n\r\n' in buffer_1:
            if len(buffer_1.split(b"\r\n\r\n")) > 2:
                print("there is a problem..")
                continue
            else:
                header, img_part = buffer_1.split(b"\r\n\r\n")
                img_length = int(header.split(b"Content-length: ")[1])
                print("img length is {}".format(img_length))
                while len(img_part) != img_length:
                    buffer_1 = tcpCliSock_1.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组12321
                    img_part += buffer_1
                print("len(img_aprt) is {}".format(len(img_part)))
                nparr = np.fromstring(img_part, np.uint8)
                print("nparr shape is {}".format(nparr.shape))
                img_decode_1 = cv2.imdecode(nparr, -1)
                cv2.imshow("img_decode_1", img_decode_1)

        # buffer_2 = tcpCliSock_2.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组
        # if b'\r\n\r\n' in buffer_2:
        #     if len(buffer_2.split(b"\r\n\r\n")) > 2:
        #         continue
        #     else:
        #         start = time.time()
        #         header, img_part = buffer_2.split(b"\r\n\r\n")
        #         img_length = int(header.split(b"Content-length: ")[1])
        #         while len(img_part) != img_length:
        #             buffer_2 = tcpCliSock_2.recv(BUFSIZ)  # 接收回应消息，接收到的是字节数组12321
        #             img_part += buffer_2
        #         nparr = np.fromstring(img_part, np.uint8)
        #         if img_length == len(nparr):
        #             print("nparr shape is {}".format(nparr.shape))
        #             img_decode_2 = cv2.imdecode(nparr, -1)
        #             print("shape is {}".format(img_decode_2.shape))
        #             cv2.imshow("img_decode_2", img_decode_2)
        #         end = time.time()
        #         print("fps: {}".format(1/(end - start)))

        cv2.waitKey(1)

        # tcpCliSock_2.close()  # 关闭客户端socket
