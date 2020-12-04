# import cv2
# import glob
# import numpy as np



def bbox_for_deepsort(*x1y1x2y2):
    '''
    return bbox info for deep_sort
    :param x1y1x2y2: (tensor(), tensor(), tensor(), tensor())
    :return: (center_x, center_y, bbox_width, bbox_height)
    '''
    x1, y1, x2, y2 = \
        x1y1x2y2[0].item(), x1y1x2y2[1].item(), x1y1x2y2[2].item(), x1y1x2y2[3].item()
    bbox_left = min(x1, x2)
    bbox_top = min(y1, y2)
    bbox_width = abs(x1 - x2)
    bbox_height = abs(y1 - y2)
    center_x = bbox_left + bbox_width / 2
    center_y = bbox_top + bbox_height / 2
    return center_x, center_y, bbox_width, bbox_height


def id_rgb(idx):
    '''
    return unique rgb corresponding to id
    :param idx: integer
    :return: rgb
    '''
    chs = (pow(2, 10) + 1, pow(2, 15) + 1, pow(2, 20) + 1)
    rgb = tuple(ch * (pow(idx + 10, 10) + 1) % 255 for ch in chs)
    return rgb


def im_trim(img, x, y, w, h):
    img_trim = img[y:y+h, x:x+w]
    return img_trim





def bbox_trim(img, bboxes, ids=None, store=False):
    '''
    trim bbox(es) on img (frame)
    :param img:
    :param bboxes:
    :param ids:
    '''
    import time

    result = {}
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        id_ = int(ids[i]) if ids is not None else 0
        min_x, min_y = x1 if x1 < x2 else x2, y1 if y1 < y2 else y2
        width, height = int(abs(x1-x2)), int(abs(y1-y2))

        if width < 32 or height < 32: # contest condition.
            continue

        img_trim = img[min_y: min_y + height, min_x: min_x + width]

        if store: # for debug
            pass
            # caption = '{:d}'.format(id_)
            # text_size, _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_PLAIN,
            #                                fontScale=2, thickness=2)
            # cv2.putText(img=img_trim, text=caption,
            #             org=(0, 0 + text_size[1] + 4), fontFace=cv2.FONT_HERSHEY_PLAIN,
            #             fontScale=2, color=(255, 255, 255), thickness=2)
            # cv2.imwrite('chanchanchan/{}-{}.jpg'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), id_), img_trim)

        result[id_] = img_trim
        # show_img2(img_trim)

    return result


# def bbox_draw(img, bboxes, ids=None):
#     '''
#     draw bbox(es) on img (frame)
#     :param img:
#     :param bboxes:
#     :param ids:
#     '''
#     for i, bbox in enumerate(bboxes):
#         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
#
#         id_ = int(ids[i]) if ids is not None else 0
#         label = ''
#         # caption (eg) label='person', id=1 → caption='person1'
#         caption = '{}{:d}'.format(label, id_)
#         text_size, _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_PLAIN,
#                                        fontScale=2, thickness=2)
#         color = id_rgb(id_)
#         # draw a bounding box(bbox)
#         cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2),
#                       color=color, thickness=3)
#         # draw a caption bar and put caption on it
#         cv2.rectangle(img=img, pt1=(x1, y1),
#                       pt2=(x1 + text_size[0] + 3, y1 + text_size[1] + 4),
#                       color=color, thickness=-1)
#         cv2.putText(img=img, text=caption,
#                     org=(x1, y1 + text_size[1] + 4), fontFace=cv2.FONT_HERSHEY_PLAIN,
#                     fontScale=2, color=(255, 255, 255), thickness=2)
#     return img
#
#
# def img_to_video(img_dir, img_type='jpg', vid_type='mp4'):
#     imgs = []
#     img_files_list = glob.glob('{}/*.{}'.format(img_dir, img_type))
#     img_files_list.sort()
#
#     for filename in img_files_list:
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width, height)
#         imgs.append(img)
#
#     out = cv2.VideoWriter('result.{}'.format(vid_type), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
#     for img in imgs:
#         out.write(img)
#     out.release()

