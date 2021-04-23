import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# haar cascade used to identify faces
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# model trained in model.py
model = load_model("mask_classification_model.h5")

video_capture = cv2.VideoCapture(0)

while True:
    # capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    faces_list = []
    predictions = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)

        if len(faces_list) > 0:
            predictions = model.predict(faces_list)

        for pred in predictions:
            (mask, withoutMask) = pred
            label = "Mask!" if mask > withoutMask else "No Mask :("
            color = (0, 255, 0) if label == "Mask!" else (0, 0, 255) # bgr colors
            label = "{} {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # display frame
    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # q to quit
        break

video_capture.release()
cv2.destroyAllWindows()
