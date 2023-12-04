import re
import errno
import os
import pandas as pd
import random
import numpy as np
import torch
import json
from PIL import Image, ImageDraw


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

def draw_box(sample, locations, phrases):
    draw = ImageDraw.Draw(sample)  
    w_img, h_img = sample.size
    for i in range(len(locations)):
        x0, y0, w, h = locations[i]
        x1, y1 = x0 + w, y0 + h
        x0, y0, x1, y1 =  x0 * w_img, y0 * h_img, x1 * w_img, y1 * h_img # (x0,y0) upper left，（x1,y1）low right
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=4)
        draw.text((x0, y0-18), phrases[i], (255, 0, 0))
    return sample


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)

def extract_prediction(input_string):
    # Regular expression pattern to match category and bounding box
    pattern = r'\b(\w+\s*\w*)\s*:\s*\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)

    # Extract categories and bounding boxes from the matches
    categories = []
    bboxes = []
    for match in matches:
        categories.append(match[0])
        bboxes.append([float(match[1]), float(match[2]), float(match[3]), float(match[4])])

    # Return the extracted categories and bounding boxes
    return categories, bboxes

def center2lefttop(boxes):
    new_boxes = []
    for jj in range(len(boxes)):
        xc, yc, w, h = boxes[jj]
        box = [xc - w/2, yc - h/2, w, h]
        new_boxes.append(box)
    return new_boxes
