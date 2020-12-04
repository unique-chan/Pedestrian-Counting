# import argparse
import sys

from util import *
from detection.yolov5.models.experimental import *
from detection.yolov5.utils.datasets import *
from detection.yolov5.utils.utils import *
from detection.yolov5.utils.torch_utils import *

from tracking.deep_sort.utils.parser import get_config
from tracking.deep_sort.deep_sort import DeepSort

import json
# import pickle


def update_json(new_id, people, json_path="./t1_res_mlv_gist.json", key="track1_results",
                id_min=0, id_max=499):
    result_dic = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            result_dic = json.load(json_file)
    else:
        result_dic[key] = [{"id": i, "people": 10} for i in range(id_min, id_max + 1)]

    # result_dic[key].append(id_people_dic)
    result_dic[key][new_id]["people"] = people

    with open(json_path, 'w') as json_file:
        json.dump(result_dic, json_file)


def count(source, clip_id, save_img=False):
    output, weights, view_img, img_size, no_save_result = \
        '.', 'detection/yolov5/weights/best.pt', False, 640, True
        # '.', 'detection/yolov5/weights/yolov5l.pt', False, 640, True

    # init yolo
    device = select_device('0')
    yolo = torch.load(weights, map_location=device)['model'].float()
    half = device.type != 'cpu'
    if half:
        yolo.half()
    img_size = check_img_size(img_size, s=yolo.stride.max())
    yolo.to(device).eval()

    source_dirs = os.listdir(source)
    source_dirs.sort()

    deep_sort_cfg = get_config()
    deep_sort_cfg.merge_from_file('./tracking/deep_sort/configs/deep_sort.yaml')

    id_trim_dics = {'cam1': None, 'cam2': None, 'cam3': None}
    for cam_no, source_dir in enumerate(source_dirs):
        # cfg.merge_from_file(args.config_deepsort)
        deepsort = DeepSort(deep_sort_cfg.DEEPSORT.REID_CKPT,
                            max_dist=deep_sort_cfg.DEEPSORT.MAX_DIST,
                            min_confidence=deep_sort_cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=deep_sort_cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=deep_sort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=deep_sort_cfg.DEEPSORT.MAX_AGE,
                            n_init=deep_sort_cfg.DEEPSORT.N_INIT,
                            nn_budget=deep_sort_cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

        dataset = LoadImages(os.path.join(source, source_dir), img_size=img_size)

        id_set = set([])
        id_trim_dic = {}
        cam_key = 'cam{}'.format(cam_no + 1)

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = yolo(img, augment=True)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5,
                                       classes=[0], agnostic=True)

            # track object with deepsort
            for i, det in enumerate(pred):
                if det is not None and len(det) > 0:
                    # rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    bboxes, confs = [], []
                    for *x1y1x2y2, conf, cls in det:
                        img_h, img_w, _ = im0s.shape
                        x_c, y_c, bbox_w, bbox_h = bbox_for_deepsort(*x1y1x2y2)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bboxes.append(obj)
                        confs.append([conf.item()])
                    bboxes, confs = torch.Tensor(bboxes), torch.Tensor(confs)

                    # multi-object-tracking: detection result -> deepsort
                    trackings = deepsort.update(bboxes, confs, im0s)

                    if trackings is not None and len(trackings) > 0:
                        bbox_xyxy = trackings[:, :4]
                        ids = trackings[:, -1]
                        new_ids = list(set(ids) - id_set)
                        new_ids.sort()

                        if new_ids:
                            new_ids_idxs = [i for i, key in enumerate(new_ids)]
                            frame_id_trim_dic = bbox_trim(im0s, bbox_xyxy[new_ids_idxs], new_ids, store=False)

                            id_set.update(frame_id_trim_dic.keys())
                            id_trim_dic.update(frame_id_trim_dic)

                    # streaming
                    if view_img:
                        cv2.imshow('streaming', im0s)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

            id_trim_dics[cam_key] = id_trim_dic

    update_json(clip_id, int(sum([len(trim) for trim in id_trim_dics.values()]) // 1.6))


if __name__ == '__main__':
    # for Yolo-v5
    sys.path.insert(0, './detection/yolov5')

    data_path = sys.argv[1]
    clips = os.listdir(data_path)
    clips.sort()
    # print('*', clips)

    with torch.no_grad():
        for clip in clips:
            clip_id = int(clip.split('_')[-1])
            count(os.path.join(data_path, clip), clip_id)
