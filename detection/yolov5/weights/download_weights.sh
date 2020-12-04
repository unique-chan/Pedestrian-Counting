#!/bin/bash
# Download common models

python -c "
# yechan added. (below 2 lines)
import os
os.chdir('..')
from utils.google_utils import *;
attempt_download('weights/yolov5s.pt');
attempt_download('weights/yolov5m.pt');
attempt_download('weights/yolov5l.pt');
attempt_download('weights/yolov5x.pt')
"
