# This is a Hand Recognition Demo


import PySimpleGUI as g
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import ImageTk
from PIL import Image


model = tf.keras.models.load_model("C:\\model")
img_path = "C:\\pictures"
result_path = "C:\\test_results"


df_results = pd.DataFrame(columns=["Image", "Prediction", "Confidence", "Correct"])


def GetPrediction(model, img, img_width=64, img_height=64):
    img = cv2.resize(img, (img_width, img_height))
    img = tf.image.rgb_to_grayscale(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    prediction_choice = prediction.argmax()
    prediction_confidence = prediction.max()
    return (prediction_choice, prediction_confidence)


def TranslatePrediction(prediction):
    if prediction == 0:  # Hold still option
        return "Open Palm"
    if prediction == 1:
        return "Down"
    if prediction == 2:
        return "Left"
    elif prediction == 3:
        return "Right"
    if prediction == 4:
        return "Up"


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

image_counter = 0

with open("C:\\image_counter.txt", "r") as f:
    image_counter = int(f.readline())

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
        prediction_values = GetPrediction(model, frame)
        cv2.imwrite(f"{img_path}\\{image_counter}.png", frame)
        window["text"].update(
            f"Prediction: {TranslatePrediction(prediction_values[0])} with confidence {round(prediction_values[1],2)}%."
        )

    if event == "yes":
        window["detect"].update(disabled=False)
        window["yes"].update(disabled=True)
        window["no"].update(disabled=True)
        window["text"].update("Prediction:")
        df_results = df_results.append(
            {
                "Image": f"{image_counter}.png",
                "Prediction": prediction_values[0],
                "Confidence": prediction_values[1],
                "Correct": 1,
            },
            ignore_index=True,
        )
        image_counter += 1

    if event == "no":
        window["detect"].update(disabled=False)
        window["yes"].update(disabled=True)
        window["no"].update(disabled=True)
        window["text"].update("Prediction:")
        df_results = df_results.append(
            {
                "Image": f"{image_counter}.png",
                "Prediction": prediction_values[0],
                "Confidence": prediction_values[1],
                "Correct": 1,
            },
            ignore_index=True,
        )
        image_counter += 1

with open("C:\\image_counter.txt", "w") as f:
    f.write(str(image_counter))

df_results.to_csv(f"{result_path}\\df.csv")

cap.release()
window.close()
