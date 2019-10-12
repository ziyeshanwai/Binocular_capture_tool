import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import os


def get_AB(Ml, Mr, ul, vl, ur, vr):
    """

    :param Ml: 左侧相机矩阵
    :param Mr: 右侧相机矩阵
    :param ul: 左侧相机图片的对应点坐标x
    :param vl: 左侧相机图片的对应点坐标y
    :param ur: 右侧相机图片的对应点坐标x
    :param vr: 右侧相机图片的对应点坐标y
    :return: A,B (4, 3) (4, 1)
    """
    A = np.array([[Ml[0, 0] - Ml[2, 0] * ul, Ml[0, 1] - Ml[2, 1] * ul, Ml[0, 2] - Ml[2, 2] * ul],
                  [Ml[1, 0] - Ml[2, 0] * vl, Ml[1, 1] - Ml[2, 1] * vl, Ml[1, 2] - Ml[2, 2] * vl],
                  [Mr[0, 0] - Mr[2, 0] * ur, Mr[0, 1] - Mr[2, 1] * ur, Mr[0, 2] - Mr[2, 2] * ur],
                  [Mr[1, 0] - Mr[2, 0] * vr, Mr[1, 1] - Mr[2, 1] * vr, Mr[1, 2] - Mr[2, 2] * vr],
                  ], dtype=np.float32)

    B = np.array([[Ml[2, 3] * ul - Ml[0, 3]], [Ml[2, 3] * vl - Ml[1, 3]],
                  [Mr[2, 3] * ur - Mr[0, 3]], [Mr[2, 3] * vr - Mr[1, 3]]], dtype=np.float32)
    return A, B


def My_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list):
    """

    :param
    Ml: 左投影矩阵
    :param
    Mr: 右侧投影矩阵
    :param
    left_uvs_list: 左侧图像像素坐标
    :param
    right_uvs_list: 右侧图像像素坐标
    :return:
    """
    x_ = []
    for i in range(0, len(left_uvs_list)):
        # A, b = get_AB(Ml, Mr, left_uvs_list[0][0], left_uvs_list[0][1], right_uvs_list[0][0], right_uvs_list[0][1])
        A_tmp, b_tmp = get_AB(Ml, Mr, left_uvs_list[i][0], left_uvs_list[i][1], right_uvs_list[i][0],
                              right_uvs_list[i][1])
        retval, X = cv2.solve(A_tmp, b_tmp, flags=cv2.DECOMP_SVD)
        # print("error is []".format(A_tmp.dot(X) - b_tmp))
        x_.append(X.ravel())
    return x_


def cv_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list):
    """
    cv2 自带的求解3D点
    :param Ml: 左投影矩阵
    :param Mr: 右侧投影矩阵
    :param left_uvs_list: 左侧图像像素坐标
    :param right_uvs_list: 右侧图像像素坐标
    :return: 计算出来的3D坐标
    """
    x_ = cv2.triangulatePoints(Ml, Mr, np.array(left_uvs_list, dtype=np.float32).T,
                               np.array(right_uvs_list, dtype=np.float32).T)
    x_ = x_.T/(x_.T[:, -1][:, np.newaxis])
    x_ = x_[:, :-1]
    return x_


def caculate_camera_Mat(pts1, pts2):
    """
    计算相机矩阵 这段函数代码有错误 不要使用
    :param pts1:
    :param pts2:
    :return: 做鱼连个相机矩阵
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Now decompose F to R and t using SVD
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    w, u, vt = cv2.SVDecomp(F)
    R = u * W * vt
    t = u[:, 2, np.newaxis]

    P1 = np.hstack((R, t))  # Projection matrix of second cam is ready

    P0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])  # Projection matrix of first cam at origin
    return P0, P1


def draw_3d_points(points_3d):
    """
    绘制3d点
    :param points_3d: numpy file
    :return:
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    plt.show()


def projection(M, points_3d):
    """
    M:投影矩阵
    x_: 3D 点 一个
    根据投影矩阵和3D点计算平面2d点
    :return:
    """
    xw = np.append(points_3d, 1)
    uv = Ml.dot(xw[:, np.newaxis])
    uv = uv / uv[2]
    return uv


def show_uv_on_img(img, uv, i):
    """
    将uv坐标点绘制在图片上
    :param img: 输入的图片numpy
    :param uv: uv坐标
    :return: 图片
    """
    cv2.circle(img, (int(uv[0]), int(uv[1])), radius, (255, 0, 0), 1)
    cv2.putText(img, "{}".format(i), (int(uvs[0] - radius / 2), int(uvs[1] + radius / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), lineType=cv2.LINE_AA)
    return img



if __name__ == "__main__":
    left_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\left"
    right_img_root_path = r"\\192.168.20.63\ai\Liyou_wang_data\double_cameras_video\imgs\right"
    left_img = cv2.imread(os.path.join(left_img_root_path, "600.jpg"))
    right_img = cv2.imread(os.path.join(right_img_root_path, "600.jpg"))
    radius = 10
    Ml = np.array([[640, 0, 360, 0], [0, 640, 640, 0], [0, 0, 1, 0]], dtype=np.float32)
    Mr = np.array([[717.53974514709, 120.3223062855059, 99.29379005953001, 685.6673689524829],
                   [126.784735664511, 647.4139422422037, 619.6618579458386, 102.889662870855],
                   [0.3686040249587207, 0.02700029362962496, 0.9291943052602857, 0.1461300101337688]], dtype=np.float32)  # 右侧相机矩阵
    left_uvs_list = [[150,178],[198,163],[276,164],[364,171],[459,177],[142,237],[510,258],[315,279],[157,375],[490,390],[275,495],[212,623],[238,605],[270,618],[304,613],[219,718],[265,747],[322,736],[190,773],[279,827],[158,466],[150,531],[145,600],[467,488],[470,560],[471,635]]

    right_uvs_list = [[243,198],[320,189],[406,198],[483,211],[557,225],[261,267],[603,303],[448,318],[271,421],[579,424],[499,539],[405,682],[444,656],[485,665],[505,655],[414,778],[467,796],[512,772],[322,849],[446,873],[287,523],[285,594],[272,673],[572,514],[581,582],[577,649]]
    # Ml, Mr = caculate_camera_Mat(np.array(left_uvs_list, dtype=np.float32), np.array(right_uvs_list, dtype=np.float32))

    x_ = np.array(My_triangulatePoints(Ml, Mr, left_uvs_list, right_uvs_list), dtype=np.float32)  # my own 3d restruction code

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, len(x_)):
        # xw = np.append(x_[i], 1)
        # uvs = Ml.dot(xw[:, np.newaxis])
        # uvs = uvs/uvs[2]
        uvs = projection(Ml, x_[i])
        left_img = show_uv_on_img(left_img, uvs, i)
        # cv2.circle(left_img, (int(uvs[0]), int(uvs[1])), radius, (255, 0, 0), 1)
        # cv2.putText(left_img, "{}".format(i), (int(uvs[0] - radius / 2), int(uvs[1] + radius / 2)), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), lineType=cv2.LINE_AA)
        print("left:{} {}".format(i, uvs))
        # uvs = Mr.dot(xw[:, np.newaxis])
        # uvs = uvs / uvs[2]
        uvs = projection(Mr, x_[i])
        right_img = show_uv_on_img(right_img, uvs, i)
        # cv2.circle(right_img, (int(uvs[0]), int(uvs[1])), radius, (255, 0, 0), 1)
        # cv2.putText(right_img, "{}".format(i), (int(uvs[0] - radius / 2), int(uvs[1] + radius / 2)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), lineType=cv2.LINE_AA)
        print("right:{} {}".format(i, uvs))
    cv2.imshow('left_image', left_img)
    cv2.imshow('right_image', right_img)
    # cv2.waitKey(0)
    # visulize face
    ax.scatter(x_[:, 0], x_[:, 1], x_[:, 2])
    print(x_)
    ax.plot(x_[[0, 1, 2, 3, 4], 0], x_[[0, 1, 2, 3, 4], 1], x_[[0, 1, 2, 3, 4], 2], color='r')
    ax.plot(x_[[5, 7, 6], 0], x_[[5, 7, 6], 1], x_[[5, 7, 6], 2], color='r')
    ax.plot(x_[[8, 20, 21, 22], 0], x_[[8, 20, 21, 22], 1], x_[[8, 20, 21, 22], 2], color='r')
    ax.plot(x_[[9, 23, 24, 25], 0], x_[[9, 23, 24, 25], 1], x_[[9, 23, 24, 25], 2], color='r')
    ax.plot(x_[[11, 12, 13, 14], 0], x_[[11, 12, 13, 14], 1], x_[[11, 12, 13, 14], 2], color='r')
    ax.plot(x_[[18, 15, 16, 17], 0], x_[[18, 15, 16, 17], 1], x_[[18, 15, 16, 17], 2], color='r')

    plt.show()

