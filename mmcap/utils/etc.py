import os.path as osp
import glob
import math
from datetime import date, datetime, timedelta

import cv2


BASE_TIME_LIST = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']
API_PROVIDED = ['0210', '0510', '0810', '1110', '1410', '1710', '2010', '2310']


def get_curr_time():
    """Get current time for weather API

    """
    today = date.today()
    yesterday = date.today() - timedelta(1)
    now = datetime.now()
    today = today.strftime('%Y%m%d')
    yesterday = yesterday.strftime('%Y%m%d')
    now = now.strftime('%H%M')

    base_date, base_time = yesterday, '2300' 

    for base_t, api_t in zip(BASE_TIME_LIST, API_PROVIDED):
        if now > api_t: 
            base_date = today
            base_time = base_t

    return base_date, base_time


def isnan(value):
    try:
        return math.isnan(float(value))
    except:
        return False


def extract_imgs_from_video(video_filepath:str, cap_locs:list):
    """Extract and Save imgs from given video file path
    Args:
        video_filepath (str)
        cap_locs (list) : frame locations list to extract
            ex) [0.25, 0.5, 0.75]
    """
    assert isinstance(cap_locs, list)

    video_id = osp.basename(video_filepath.split('.')[0])
    video_dir = '/'.join(video_filepath.split('/')[:-1])

    cap = cv2.VideoCapture(video_filepath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap_frames = [int(video_length * loc) for loc in cap_locs]

    img_path_list = list()
    for frame in cap_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        save_fn = '{0}_{1:0>5}.jpg'.format(video_id, str(frame))
        save_path = osp.join(video_dir, save_fn)
        cv2.imwrite(save_path, img)
        img_path_list.append(save_path)

    return img_path_list

