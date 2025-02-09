import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm


click_coords = []


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_copy = img.copy()
        xy = f"({x}, {y})"
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), thickness=-1)
        cv2.putText(img_copy, xy, (x + 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=1)
        cv2.imshow("image", img_copy)
        click_coords.append((x, y))


def get_coordinate_by_click(frame):
    global img
    img = frame
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("image", 150, 150)
    cv2.setMouseCallback("image", mouse)
    cv2.imshow("image", img)
    while len(click_coords) < 2:
        if cv2.getWindowProperty("image", 0) == -1:
            break
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return click_coords


def time_string_to_seconds(time_string):
    if time_string is None:
        return None
    time_format = "%H:%M:%S"
    time_obj = datetime.strptime(time_string, time_format)
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds


def calc_similarity(img1, img2):
    # 转换为HSV并计算直方图
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # 归一化并比较直方图（值越接近1越相似）
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def extract_key_frames(video_path, start_time=None, end_time=None, top_left=None, bottom_right=None, threshold=0.999):
    output_dir = video_path + ".frames"

    start_time = time_string_to_seconds(start_time)
    end_time = time_string_to_seconds(end_time)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    index = 0
    previous_frame = None
    similarity_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if index == 0:
            click_coords = get_coordinate_by_click(frame)
            if top_left is None or bottom_right is None:
                top_left, bottom_right = click_coords

        index += 1

        if end_time and cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time * 1000:
            break

        if top_left and bottom_right:
            frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        if previous_frame is not None:
            similarity = calc_similarity(previous_frame, frame)
            similarity_list.append(similarity)

            if similarity > threshold:
                continue

            print(f"index: {index}, similarity: {similarity}")

        cv2.imwrite(f"{output_dir}/{index}.jpg", frame)
        previous_frame = frame

    cap.release()
    cv2.destroyAllWindows()

    # import matplotlib.pyplot as plt
    # plt.plot(similarity_list, marker="o")
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    video_path = "/path/to/video.mp4"
    start_time = "00:00:00"
    end_time = None
    extract_key_frames(video_path, start_time, end_time)
