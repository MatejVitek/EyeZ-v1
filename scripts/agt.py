#!/usr/bin/env python3
import cv2
from joblib import Parallel, delayed
import os
import time


def process_file(f):
	img = cv2.imread(f)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
	img = cv2.medianBlur(img, 5)
	cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


n = 100
dir = os.path.join('/hdd', 'EyeZ', 'SBVPI', 'stage2')
parallel = False

start = time.time()
if parallel:
	Parallel(-1)(delayed(process_file)(os.path.join(dir, f)) for f in os.listdir(dir)[:n])
else:
	for f in os.listdir(dir)[:n]:
		process_file(os.path.join(dir, f))
print(f"Time per image: {time.time() - start}s")
