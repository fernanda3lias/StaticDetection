import cv2
import numpy as np
from collections import Counter, defaultdict

#import file from path
BACKGROUND_PATH = r"background.png"
VIDEO_PATH = r"train.mp4"

consecutiveframe = 20
track_temp = []
track_master = []
track_temp2 = []

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

frameno = 0
#counter=1






