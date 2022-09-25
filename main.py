# This is a Hand Recognition Demo


import PySimpleGUI as g
import cv2


g.change_look_and_feel("Dark")


btn_color = "black on yellow"
frame_font = text_font = btn_font = "Inter"
btn_size = (25, 2)


col1 = [
    [
        g.Frame(
            "Choices",
            [
                [
                    g.Button(
                        "Turn On Camera ON/OFF",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                    )
                ],
                [
                    g.Button(
                        "Detect Hand ON/OFF",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                    )
                ],
                [
                    g.Button(
                        "Take Picture",
                        button_color=btn_color,
                        size=btn_size,
                        font=btn_font,
                    )
                ],
                [
                    g.Button(
                        "Predict Visual",
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
            size=(800, 600),
            font=frame_font,
            border_width=3,
        )
    ]
]


layout = [
    [
        g.Column(col1, element_justification="c"),
        g.Column(col2, element_justification="c"),
    ],
]


window = g.Window("Hand Recognition Demo", layout)


while True:
    event, values = window.read()
    if event == "Ex1" or event == g.WIN_CLOSED:
        break


window.close()
