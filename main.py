# This is a Hand Recognition Demo

import PySimpleGUI as g


g.change_look_and_feel('Dark')


btn_color = 'black on yellow'
btn_font = 'Inter'
btn_size = (20,2)

wind_size = (800, 600)

col1 = [[g.Button('Turn On Camera', button_color = btn_color, size = btn_size , font = btn_font)],
            [g.Button('Detect Hand', button_color = btn_color, size = btn_size, font = btn_font)],
            [g.Button('Take Picture', button_color = btn_color, size = btn_size, font = btn_font)],
            [g.Button('Predict Visual', button_color = btn_color, size = btn_size, font = btn_font)]]

layout = [[g.Column(col1)]]


window = g.Window("Hand Recognition Demo", layout, size = wind_size)


while True:
    event, values = window.read()
    if event == "Ex1" or event == g.WIN_CLOSED:
        break
window.close()