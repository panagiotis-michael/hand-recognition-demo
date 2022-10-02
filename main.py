# This is a Hand Recognition Demo


import PySimpleGUI as g
import cv2
import pandas as pd

# import tensorflow as tf


# model = tf.keras.models.load_model(
# "C:\\Users\\bignu\\OneDrive\\Desktop\\Hand Recognition Demo\\hand-recognition-demo\\model"
# )


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
                    )
                ],
                [
                    g.Button(
                        "Detect Hand",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
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
        )
    ],
    [
        g.Button(
            "YES",
            button_color=btn_color2,
            size=btn_size,
            font=btn_font,
        ),
        g.Button(
            "NO",
            button_color=btn_color3,
            size=btn_size,
            font=btn_font,
        ),
    ],
]


layout = [
    [
        g.Column(col1, element_justification="c"),
        g.Column(col2, element_justification="c"),
    ],
]


window = g.Window("Hand Recognition Demo", layout, keep_on_top=True)


while True:
    event, values = window.read()
    if event == "Ex1" or event == g.WIN_CLOSED:
        break


window.close()
