import cv2
import numpy as np
from keras_applications import inception_v3
from tensorflow import keras

model = keras.models.load_model("model.h5")

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera_height = 500

while (True):
    # read a new frame
    _, frame = camera.read()

    # flip the frameq
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    #for ip webcam
    #aspect = 16/9
    res = int(aspect * camera_height)  # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))
    #for ip webcam


    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)

    # get ROI
    roi = frame[75 + 2:425 - 2, 300 + 2:650 - 2]

    # parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # resize to 224*224
    roi = cv2.resize(roi, (150, 150))


    # predict!
    roi2 = np.array([cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)])

    predictions = model.predict(roi2)

    if predictions[0][0] ==1 :
        labels ="dog"
    else :
        labels = "cat"

    # add text
    labels = cv2.putText(frame, labels, (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)



    cv2.imshow("Real Time object detection", frame)

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()