#!/usr/bin/env python3
import cv2
from joblib import Parallel, delayed
import os
import time


def process_file(f):
	if os.path.splitext(f)[1].lower() != '.jpg':
		return
	print(f)
	img = cv2.imread(f)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
	img = cv2.medianBlur(img, 3)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
	cv2.imwrite('./agt/' + os.path.basename(f), img)


dir = os.path.join('/hdd', 'EyeZ', 'SBVPI', 'Subsets', 'Rot ScleraNet', 'stage2_ungrouped')
Parallel(-1)(delayed(process_file)(os.path.join(dir, f)) for f in os.listdir(dir))

