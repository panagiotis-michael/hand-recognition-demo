# This is a Hand Recognition Demo


import PySimpleGUI as g
import cv2
import pandas as pd
import numpy as np
#import tensorflow as tf
from PIL import ImageTk
from PIL import Image


# model = tf.keras.models.load_model(
# "C:\\Users\\bignu\\OneDrive\\Desktop\\Hand Recognition Demo\\hand-recognition-demo\\model"
# )


def GetPrediction(model, img, img_width=64, img_height=64):
    img = cv2.resize(img, (img_width, img_height))
    img = tf.image.rgb_to_grayscale(img)
    img_array = img.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    prediction_choice = prediction.argmax()
    prediction_confidence = prediction.max()
    return (prediction_choice, prediction_confidence)


g.change_look_and_feel("Dark2")


btn_color = "black on yellow"
btn_color2 = "green on yellow"
btn_color3 = "red on yellow"
text_font = frame_font = text_font = btn_font = "Inter"
btn_size = (25, 2)


col = [
    [
        g.Frame(
            "Camera",
            [[g.Image(filename="", key="camera")]],
            # size=(900, 600),
            font=frame_font,
            border_width=3,
        )
    ],
    [
        g.Text(
            "Prediction: ",
            size=(60, 2),
            font=text_font,
            text_color="#FFFFFF",
            justification="c",
            key="text",
        )
    ],
    [
        g.Button(
            "Detect Hand",
            button_color=btn_color,
            size=btn_size,
            font=btn_font,
            key="detect",
        ),
        g.Button(
            "YES",
            button_color=btn_color2,
            size=btn_size,
            font=btn_font,
            key="yes",
            disabled=True,
        ),
        g.Button(
            "NO",
            button_color=btn_color3,
            size=btn_size,
            font=btn_font,
            key="no",
            disabled=True,
        ),
    ],
]


layout = [
    [g.Column(col, element_justification="c")],
]


window = g.Window(
    "Hand Recognition Demo",
    layout,
    keep_on_top=True,
)

cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=30)

    if event == g.WIN_CLOSED:
        break

    ret, frame = cap.read()
    imgbytes = ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    window["camera"].update(data=imgbytes)

    if event == "detect":
        window["detect"].update(disabled=True)
        window["yes"].update(disabled=False)
        window["no"].update(disabled=False)
        prediction_values = GetPrediction(model,frame)
        window["text"].update(f"Prediction: {prediction_values[0]} with confidence {prediction_values[1]}%.")

    if event == "yes":
        window["detect"].update(disabled=True)
        window["yes"].update(disabled=True)
        window["no"].update(disabled=True)
        window["text"].update("Prediction:")

    if event == "no":
        window["detect"].update(disabled=True)
        window["yes"].update(disabled=True)
        window["no"].update(disabled=True)
        window["text"].update("Prediction:")
    
    
cap.release()
window.close()
