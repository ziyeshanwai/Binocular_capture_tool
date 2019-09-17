import cv2
import numpy as np
import os


i = 0
radius = 10
coordinate_list = []
# mouse callback function


def draw_circle(event, x, y, flags, param):
    global coordinate_list, i
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), radius, (255, 0, 0), 1)
        cv2.putText(img, "{}".format(i), (int(x-radius/2), int(y+radius/2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), lineType=cv2.LINE_AA)
        print("x is {}, y is {}".format(x, y))
        coordinate_list.append([x, y])
        i += 1


if __name__ == "__main__":
    img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\left"
    img = cv2.imread(os.path.join(img_root_path, "0.jpg"))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    camera = "left"
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord('p'):
            print("{}".format(coordinate_list))
            with open("cor-{}.txt".format(camera), "w") as f:
                for i in range(0, len(coordinate_list)):
                    f.write("{{{},{}}},".format(coordinate_list[i][0], coordinate_list[i][1]))
    cv2.destroyAllWindows()
