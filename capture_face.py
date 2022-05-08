import numpy as np
import dlib
import cv2
import tensorflow as tf
import tensorflow_addons as tfa

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
fbeta = tfa.metrics.FBetaScore(beta=1.0, num_classes=len(class_names), average='macro')
model = tf.keras.models.load_model('model/Model.126-0.6333.hdf5', custom_objects={"FBetaScore": fbeta})

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    key = cv2.waitKey(1)

    for det in dets:
        y = det.top()
        yh = det.bottom()
        x = det.left()
        xw = det.right()
        crop_img = gray[y:yh, x:xw]
        resized_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_AREA).reshape((1, 48, 48, 1))
        resized_img = resized_img.astype(np.float32) / 255
        pred = np.argmax(model.predict(resized_img))
        pred_class = class_names[pred]
        # show the predicted emotion number
        cv2.putText(img, f"Emotion: {pred_class}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img,(x, y), (xw, yh), color_green, line_width)
    
    cv2.imshow('my webcam', img)

    if key == ord('q'):
        break  # q to quit

cam.release()
cv2.destroyAllWindows()