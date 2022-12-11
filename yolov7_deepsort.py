import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np



from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box



# from detector import build_detector
from deep_sort import build_tracker
from deepsort_utils.draw import draw_boxes
from deepsort_utils.parser import get_config
from deepsort_utils.log import get_logger
from deepsort_utils.io import write_results

from PIL import Image

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

gpu = select_device('')
half = gpu.type != 'cpu'
weights = 'yolov7.pt'
imgsz = 640
model = attempt_load(weights, map_location=gpu)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)
model = TracedModel(model, gpu, imgsz)
model.half()


names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def yolo_detect(source):
    result_bbox_xywh = []
    result_cls_conf = []
    result_cls_ids = []

    old_img_w = old_img_h = imgsz
    old_img_b = 1
    # source = cv2.imread('../inference/images/bus.jpg')
    img_1 = letterbox(source, imgsz, stride=stride)[0]
    # Convert
    img = img_1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(gpu)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # warm up
    if gpu.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=False)[0]
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred)
    det = pred[0]
    s, im0 = '', source
    #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        # Write results
        for *xyxy, conf, cls in reversed(det):
            # label = f'{names[int(cls)]} {conf:.2f}'
            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
            result_bbox_xywh.append(xywh)
            result_cls_conf.append(float(conf))
            result_cls_ids.append(int(cls))
    return result_bbox_xywh, result_cls_conf, result_cls_ids





class VideoTracker(object):
    def __init__(self, args, video_path):
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        # self.detector = build_detector()
        self.deepsort = build_tracker(use_cuda=True)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

 
    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            ref, ori_im = self.vdo.retrieve()

            if ref is True:
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                # #----- do detection
                # frame = Image.fromarray(np.uint8(im))
                bbox_xywh, cls_conf, cls_ids = yolo_detect(ori_im)
                if cls_conf is not None:
                    #-----copy
                    list_fin = []
                    for i in bbox_xywh:
                        temp = []
                        temp.append(i[0])
                        temp.append(i[1])
                        temp.append(i[2]*1.)
                        temp.append(i[3]*1.)
                        list_fin.append(temp)
                    new_bbox = np.array(list_fin).astype(np.float32)

                    #-----#-----mask processing filter the useless part
                    mask = [0,1,2,3,5,7]# keep specific classes the indexes are corresponded to coco_classes
                    mask_filter = []
                    for i in cls_ids:
                        if i in mask:
                            mask_filter.append(1)
                        else:
                            mask_filter.append(0)
                    new_cls_conf = []
                    new_new_bbox = []
                    new_cls_ids = []
                    for i in range(len(mask_filter)):
                        if mask_filter[i]==1:
                            new_cls_conf.append(cls_conf[i])
                            new_new_bbox.append(new_bbox[i])
                            new_cls_ids.append(cls_ids[i])
                    new_bbox =  np.array(new_new_bbox).astype(np.float32)
                    cls_conf =  np.array(new_cls_conf).astype(np.float32)
                    cls_ids  =  np.array(new_cls_ids).astype(np.float32) 
                    #-----#-----

                    # do tracking
                    outputs = self.deepsort.update(new_bbox, cls_conf, im)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_tlwh = []
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                        for bb_xyxy in bbox_xyxy:
                            bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                        results.append((idx_frame - 1, bbox_tlwh, identities))

                    end = time.time()

                    if self.args.display:
                        cv2.imshow("test", ori_im)
                        cv2.waitKey(1)

                    if self.args.save_path:
                        self.writer.write(ori_im)

                    # save results
                    write_results(self.save_results_path, results, 'mot')

                    # logging
                    self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                                    .format(end - start, 1 / (end - start), new_bbox.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with VideoTracker( args, video_path='./001.avi') as vdo_trk:
        vdo_trk.run()
