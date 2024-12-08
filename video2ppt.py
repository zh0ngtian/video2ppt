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
    time_format = "%H:%M:%S"
    time_obj = datetime.strptime(time_string, time_format)
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds


def extract_key_frames(video_path, start_time=None, end_time=None, top_left=None, bottom_right=None, threshold=0.8):
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
            diff = cv2.absdiff(previous_frame, frame)
            non_zero_count = np.count_nonzero(diff)
            total_pixels = diff.size
            
            # 计算差异比例
            diff_ratio = non_zero_count / total_pixels

            if diff_ratio < threshold:
                continue
            print(f"index: {index}, diff_ratio: {diff_ratio}")

        cv2.imwrite(f"{output_dir}/{index}.jpg", frame)
        previous_frame = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "2023_4_15_赵声良《犍陀罗与敦煌》_哔哩哔哩_bilibili.mp4"
    start_time = "00:25:00"
    end_time = "01:17:50"
    extract_key_frames(video_path, start_time, end_time)
