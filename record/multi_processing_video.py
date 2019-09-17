from threading import Thread
import cv2
import time


class VideoWriterWidget(object):
    def __init__(self):
        # Create a VideoCapture object
        self.left_capture = cv2.VideoCapture(1)
        self.right_capture = cv2.VideoCapture(0)
        self.frame_width = 1280
        self.frame_height = 720
        self.left_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.left_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.right_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.right_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.video_left_file_name = "./output/left.avi"
        self.video_right_file_name = "./output/right.avi"
        self.record = True
        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.output_left_video = cv2.VideoWriter(self.video_left_file_name, self.codec, 30,
                                                 (self.frame_width, self.frame_height))
        self.output_right_video = cv2.VideoWriter(self.video_right_file_name, self.codec, 30,
                                                 (self.frame_width, self.frame_height))
        self.exit = False
        self.left_frame_number = 0
        self.right_frame_number = 0
        # Start the thread to read frames from the video stream
        self.thread_left = Thread(target=self.update_left, args=())
        self.thread_left.daemon = True
        self.thread_left.start()

        self.thread_right = Thread(target=self.update_right, args=())
        self.thread_right.daemon = True
        self.thread_right.start()

        # Start another thread to show/save frames
        self.start_show_video()
        self.star_record()
        print("start..")

    def update_left(self):
        # Read the next frame from the stream in a different thread
        while True:
            start = time.time()
            if self.left_capture.isOpened():
                (self.left_status, self.left_frame) = self.left_capture.read()
                self.left_frame_number += 1
            end = time.time()
            print("left camera it takes {}s".format(end-start))

    def update_right(self):
        while True:
            start = time.time()
            if self.right_capture.isOpened():
                (self.right_status, self.right_frame) = self.right_capture.read()
                self.right_frame_number += 1
            end = time.time()
            print("right camera it takes {}s".format(end - start))

    def show_frame(self):
        # Display frames in main progra
        if self.left_status:
            im_left_show = cv2.transpose(cv2.resize(self.left_frame, (0, 0), fx=0.5, fy=0.5))
            cv2.imshow("capture_left", cv2.flip(im_left_show, 0))
            # cv2.imshow("left_camera", self.left_frame)
        if self.right_status:
            im_right_show = cv2.transpose(cv2.resize(self.right_frame, (0, 0), fx=0.5, fy=0.5))
            cv2.imshow("capture_right", cv2.flip(im_right_show, 0))

        print("show left frame number:{} right frame number:{}".format(self.left_frame_number, self.right_frame_number))

    def save_frame(self):
        print(self.left_frame.shape)
        # Save obtained frame into video output file
        self.output_left_video.write(self.left_frame)
        self.output_right_video.write(self.right_frame)
        print("save left frame number:{}, right frame number:{}".format(self.left_frame_number, self.right_frame_number))

    def start_show_video(self):
        # Create another thread to show/save frames
        def start_show_thread():
            while True:
                try:
                    self.show_frame()
                except AttributeError:
                    pass
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("over in record")
                    self.left_capture.release()
                    self.output_left_video.release()
                    self.right_capture.release()
                    self.output_right_video.release()
                    cv2.destroyAllWindows()
                    self.exit = True
                    exit(1)
                if cv2.waitKey(33) == ord('a'):
                    self.record = True
        self.show_thread = Thread(target=start_show_thread, args=())
        self.show_thread.daemon = True  # 守护进程
        self.show_thread.start()

    def star_record(self):
        def start_record_thread():
            while True:
                if self.record:
                    try:
                        self.save_frame()
                    except AttributeError:
                        pass
        self.recording_thread = Thread(target=start_record_thread, args=())
        self.recording_thread.daemon = True  # 守护进程
        self.recording_thread.start()


if __name__ == '__main__':
    video_writer_widget = VideoWriterWidget()
    while True:
        if video_writer_widget.exit:
            break
            # print("start recording..")
        time.sleep(0.00001)
    # video_writer_widget.show_thread.join()
    # video_writer_widget.thread.join()
    # video_writer_widget.recording_thread.join()