"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

import json
from collections import Counter

import cv2
import os
from libs.draw import Draw
from libs.geometric import get_polygon, get_line, get_box
from libs.validate import validate
from openvino.inference_engine import IENetwork, IECore


class LineMonitoring(object):
    def __init__(self):
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file_path) as f:
            cfg = json.load(f)
            validate(cfg)

        self.running = True
        self.videosource = cfg.get("video")
        self.model_modelfile = cfg.get("pedestrian_model_weights")
        self.model_configfile = cfg.get("pedestrian_model_description")
        self.coords = cfg.get("coords")
        self.totalFrames = 0
        self.results = {}
        # OPENVINO VARS
        self.ov_input_blob = None
        self.out_blob = None
        self.net = None
        self.ov_n = None
        self.ov_c = None
        self.ov_h = None
        self.ov_w = None
        # PROCESSOR VARS
        self.confidence_threshold = .65
        self.polygon = None
        self.line = None
        self.trend = []
        self.most = 0
        self.counter = Counter()
        self.trend_window = 81

    def load_openvino(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile, weights=self.model_modelfile)
            self.ov_input_blob = next(iter(net.inputs))
            self.out_blob = next(iter(net.outputs))
            self.net = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n, self.ov_c, self.ov_h, self.ov_w = net.inputs[self.ov_input_blob].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino error:{e}")

    def config_env(self, frame):
        h, w = frame.shape[:2]
        line = ((int(self.coords[0][0] * w / 100), int(self.coords[0][1] * h / 100)),
                (int(self.coords[1][0] * w / 100), int(self.coords[1][1] * h / 100)))
        line = get_line(line)
        self.line = line
        self.polygon = line.buffer(5)
        self.polygon = get_polygon(list(self.polygon.exterior.coords))

    def get_frame(self):
        h = w = None
        try:
            cap = cv2.VideoCapture(self.videosource)
        except Exception as e:
            raise Exception(f"Video source error: {e}")

        while self.running:
            has_frame, frame = cap.read()
            if has_frame:
                if w is None or h is None:
                    h, w = frame.shape[:2]
                    self.config_env(frame)
                self.totalFrames += 1
                yield frame
            else:
                self.running = False
        return None

    def process_frame(self, frame):
        _frame = frame.copy()
        if self.totalFrames % 4 == 0:
            count_people = 0

            frame = cv2.resize(frame, (self.ov_w, self.ov_h))
            frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            frame = frame.reshape((self.ov_n, self.ov_c, self.ov_h, self.ov_w))

            self.net.start_async(request_id=0, inputs={self.ov_input_blob: frame})

            if self.net.requests[0].wait(-1) == 0:
                res = self.net.requests[0].outputs[self.out_blob]

                frame = _frame
                h, w = frame.shape[:2]
                out = res[0][0]
                for i, detection in enumerate(out):

                    confidence = detection[2]
                    if confidence > self.confidence_threshold and int(detection[1]) == 1:  # 1 => CLASS Person

                        xmin = int(detection[3] * w)
                        ymin = int(detection[4] * h)
                        xmax = int(detection[5] * w)
                        ymax = int(detection[6] * h)

                        bbox = get_box((xmin, ymin, xmax, ymax))
                        if self.polygon.intersects(bbox):
                            count_people += 1
                            Draw.rectangle(frame, (xmin, ymin, xmax, ymax), "green", 2)

            self.trend.append(count_people)
        if len(self.trend) == self.trend_window:
            self.counter.update(self.trend)
            self.most = self.counter.most_common()[0][0]
            self.results.clear()
            self.results["Line Count:"] = self.most
            self.trend.clear()
            self.counter.clear()

        line = list(self.line.coords)
        Draw.line(frame, (int(line[0][0]), int(line[0][1]), int(line[1][0]), int(line[1][1])), "yellow", 3)
        Draw.data(frame, self.results)
        return frame

    def render(self, frame):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()

    def run(self):
        self.load_openvino()
        for frame in self.get_frame():
            frame = self.process_frame(frame)
            self.render(frame)


if __name__ == '__main__':
    try:
        lm= LineMonitoring()
        lm.run()
    except Exception as exception:
        print(exception)
