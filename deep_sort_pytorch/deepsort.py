import os
import cv2
import time
import argparse
import torch
import warnings
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # Adaptation: Video capture not needed since frames as images are used
        #if args.cam != -1:
        #    print("Using webcam " + str(args.cam))
        #    self.vdo = cv2.VideoCapture(args.cam)
        #else:
        #    self.vdo = cv2.VideoCapture()

        # Adaptation: Replace the video capture with loading a list of image (frame) paths
        self.video_frames = sorted([
            os.path.join(self.video_path, frame)
            for frame in os.listdir(self.video_path)
            # Allow frames in jpg, jpeg or png format
            if frame.lower().endswith('.jpg') or frame.lower().endswith('.jpeg') or frame.lower().endswith('.png')
        ])
            
        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=self.args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    # Adaptation: Function enter needed for this tracker, but remove content (apart from save_path) that relies on video or camera input that is not needed
    def __enter__(self):
        #if self.args.cam != -1:
        #    ret, frame = self.vdo.read()
        #    assert ret, "Error: Camera error"
        #    self.im_width = frame.shape[0]
        #    self.im_height = frame.shape[1]

        #else:
        #    assert os.path.isfile(self.video_path), "Path error"
        #    self.vdo.open(self.video_path)
        #    self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        #    self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #    assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # TODO save masks

            # path of saved video and results
        #    self.save_video_path = os.path.join(self.args.save_path, "results.avi")
        #    self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
        #    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #    self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
        #    self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        with open('coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)
        # Adaptation: Loop through image (frame) folder instead of actual video
        for img_path in self.video_frames:
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            # Adaptation: Load image from path, not from actual video
            ori_im = cv2.imread(img_path)
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # Adaptation: Don't filter by person only, living_cell and dead_cell are used in this project
            # select person class
            #mask = cls_ids == 0

            #bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            #bbox_xywh[:, 2:] *= 1.2
            #cls_conf = cls_conf[mask]
            #cls_ids = cls_ids[mask]

            # do tracking
            if self.args.segment:
                seg_masks = seg_masks[mask]
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                names = [idx_to_class[str(label)] for label in cls]

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, None if not self.args.segment else mask_outputs)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities, cls))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            # Adaptation: Don't write results frame by frame, but all at once after the loop for more effectiveness
            #write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))

        # Adaptation: Save results after the loop is done instead of frame-wise
        if self.args.save_path:
            write_results(self.save_results_path, results, 'mot')

def parse_args():
    parser = argparse.ArgumentParser()
    # Adaptation: Change default from demo video to first folder of dataset_jpg
    parser.add_argument("--VIDEO_PATH", type=str, default='dataset_jpg/dataset/001')
    # Adaptation: Remove config files as arguments; detection done via YOLOv5, appearance model done manually as an argument
    #parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    #parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    #parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    #parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    # Adaptation: Remove fastreid, mmdet and segment as arguments; prevents using the wrong appearance model or wrong detectors and doing unneeded segmentation for cells
    #parser.add_argument("--fastreid", action="store_true")
    #parser.add_argument("--mmdet", action="store_true")
    #parser.add_argument("--segment", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    # Adaptation: Add appearance model as an argument instead of as a YAML file; allows direct value manipulation through cell_tracking.ipynb
    parser.add_argument("--appearance_model", type=str)
    # Adaptation: Add tracking parameters of deep_sort.yaml as arguments; allows direct value manipulation through cell_tracking.ipynb
    # Defaults equal the given values from the original deep_sort.yaml file
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--max_iou_distance", type=float, default=0.7)
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--nms_max_overlap", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Adaptation: Remove loading YAML files since they are not used now
    #cfg = get_config()
    #if args.segment:
    #    cfg.USE_SEGMENT = True
    #else:
    #    cfg.USE_SEGMENT = False
    #if args.mmdet:
    #    cfg.merge_from_file(args.config_mmdetection)
    #    cfg.USE_MMDET = True
    #else:
    #    cfg.merge_from_file(args.config_detection)
    #    cfg.USE_MMDET = False
    #cfg.merge_from_file(args.config_deepsort)
    #if args.fastreid:
    #    cfg.merge_from_file(args.config_fastreid)
    #    cfg.USE_FASTREID = True
    #else:
    #    cfg.USE_FASTREID = False

    # Adaptation: Create a config class that replaces any YAML files and handles any parameters directly
    class CellTrackingCfg:
        def __init__(self, args):
            # Adaptation: Set fastreid, segment and mmdet to False; prevents using the wrong appearance model or wrong detectors and doing unneeded segmentation for cells
            self.USE_SEGMENT = False
            self.USE_MMDET = False
            self.USE_FASTREID = False
            self.DEEPSORT = type('', (), {})()
            # Adaptation: Set all tracking parameters (originally from deep_sort.yaml) to the given values from the arguments
            self.DEEPSORT.MAX_AGE = args.max_age
            self.DEEPSORT.N_INIT = args.n_init
            self.DEEPSORT.NN_BUDGET = args.nn_budget
            self.DEEPSORT.MIN_CONFIDENCE = args.min_confidence
            self.DEEPSORT.MAX_IOU_DISTANCE = args.max_iou_distance
            self.DEEPSORT.MAX_DIST = args.max_dist
            self.DEEPSORT.NMS_MAX_OVERLAP = args.nms_max_overlap
            # Adaptation: Sets cell-trained appearance model
            self.DEEPSORT.REID_CKPT = args.appearance_model

    # Adaptation: Initialize config class with the passed arguments
    cfg = CellTrackingCfg(args)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()