import time
import cv2
import numpy as np
import onnx
import utils.box_utils_numpy as box_utils
import onnxruntime as ort


class FaceDetector:
    def __init__(self, model_path, image_size=(320, 240), conf_threshold=0.7, iou_threshold=0.3, top_k=-1):
        self.model_path = model_path
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.top_k = top_k

        self.image_mean = np.array([127, 127, 127])

    def load_model(self):
        predictor = onnx.load(self.model_path)
        onnx.checker.check_model(predictor)
        onnx.helper.printable_graph(predictor.graph)

        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def preprocess(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, self.image_size)
        # image = cv2.resize(image, (640, 480))
        im = (im - self.image_mean) / 128
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, axis=0)
        im = im.astype(np.float32)
        return im

    def inference(self, image):
        im = self.preprocess(image)
        confidences, boxes = self.ort_session.run(
            None, {self.input_name: im})
        boxes, labels, probs = self.nms(image, confidences, boxes)
        bboxes = []
        for box, label, prob in zip(boxes, labels, probs):
            x1, y1, x2, y2 = map(int, box)
            bboxes.append([x1, y1, x2, y2, prob, label])
        return bboxes

    def nms(self, image, confidences, boxes):
        height, width, _ = image.shape
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.conf_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=self.iou_threshold,
                                           top_k=self.top_k,
                                           )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


if __name__ == "__main__":
    face_detector = FaceDetector(
        model_path="resources/weights/onnx/version-RFB-320.onnx")
    face_detector.load_model()

    path = 0
    cap = cv2.VideoCapture(path)
    old_time = time.time()
    count = 0
    fps = 0
    while True:
        if time.time() - old_time > 1:
            old_time = time.time()
            fps = count
            count = 0
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        bboxes = face_detector.inference(frame)
        for box in bboxes:
            x1, y1, x2, y2, prob, label = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
