import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
        )

    def detect(self, frame):
        """
        人脸检测函数
        参数：
            frame: 输入图像帧
        返回：
            faces: 检测到的人脸位置列表
            face_detected: 是否检测到人脸的标志
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        face_detected = False
        faces = []

        if results.detections:
            face_detected = True
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                faces.append(bbox)

        return faces, face_detected