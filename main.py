# This is a Hand Recognition Demo


import PySimpleGUI as g
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf


# model = tf.keras.models.load_model(
# "C:\\Users\\bignu\\OneDrive\\Desktop\\Hand Recognition Demo\\hand-recognition-demo\\model"
# )


def GetPrediction(model, img, img_width, img_height):
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


col1 = [
    [
        g.Frame(
            "Choices",
            [
                [
                    g.Button(
                        "Open Camera",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                        key="open",
                    )
                ],
                [
                    g.Button(
                        "Detect Hand",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                        key="detect",
                    )
                ],
                [
                    g.Button(
                        "Close Camera",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                        key="close",
                    )
                ],
            ],
            font=frame_font,
            border_width=3,
        )
    ]
]


col2 = [
    [
        g.Frame(
            "Camera",
            [[g.Image(filename="", key="image")]],
            size=(900, 600),
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
            key="pred",
        )
    ],
    [
        g.Button(
            "YES", button_color=btn_color2, size=btn_size, font=btn_font, key="yes"
        ),
        g.Button("NO", button_color=btn_color3, size=btn_size, font=btn_font, key="no"),
    ],
]


layout = [
    [
        g.Column(col1, element_justification="c"),
        g.Column(col2, element_justification="c"),
    ],
]


window = g.Window(
    "Hand Recognition Demo",
    layout,
    keep_on_top=True,
)


while True:
    event, values = window.read()

    if event == "Ex1" or event == g.WIN_CLOSED:
        break

    if event == "open":
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["image"].update(data=imgbytes)

    if event == "close":
        cap.release()
        break


window.close()
