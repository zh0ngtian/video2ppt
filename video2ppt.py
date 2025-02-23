import os
from datetime import datetime

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
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
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    similarity, _ = ssim(img1_gray, img2_gray, full=True)
    threshold = 0.9
    return similarity, threshold


def extract_key_frames(video_path, start_time=None, end_time=None, top_left=None, bottom_right=None):
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if end_time:
        selected_frames = int((end_time - start_time) * fps)
    else:
        selected_frames = total_frames - int(start_time * fps)

    previous_frame = None
    similarity_list = []

    for index in tqdm(range(selected_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        if index == 0:
            previous_frame = frame
            click_coords = get_coordinate_by_click(frame)
            if top_left is None or bottom_right is None:
                top_left, bottom_right = click_coords

        if top_left and bottom_right:
            frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        similarity, threshold = calc_similarity(previous_frame, frame)
        similarity_list.append(similarity)
        previous_frame = frame

        if similarity < threshold or index == 0:
            cv2.imwrite(f"{output_dir}/{index}_{similarity}.jpg", frame)

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
