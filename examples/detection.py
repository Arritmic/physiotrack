from physiotrack import Detection, Models
import cv2

image = cv2.imread('frame_1.png')
# Default Person Detector
detector = Detection.Person(model=Models.Detection.YOLO.PERSON.m_person, conf=0.25, iou=0.45)
result = detector.predict(image)   # or detector(image)
annotated = result.plot()
cv2.imwrite('out.png', annotated)
