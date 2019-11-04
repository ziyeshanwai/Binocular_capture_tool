import socket


if __name__ == "__main__":
    ip = "192.168.106.80"
    port = 8080
    request = b"GET / HTTP/1.1\nHost: 192.168.106.80/camera/mjpeg\n\n"
    # request = b"GET / HTTP/1.1\nHost: 192.168.106.80\n\n"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.send(request)
    result = s.recv(10000)
    print(result)
    while True:
        result = s.recv(1000)
        # print("\n\n\n 12334")
        print(result)