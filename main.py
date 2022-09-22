# This is a Hand Recognition Demo

import PySimpleGUI as g


g.change_look_and_feel('Dark')


bt_color = 'black on yellow'
bt_font = 'Inter'
bt_size = (20,2)


layout = [[g.Button('Turn On Camera', button_color = bt_color, size = bt_size , font = bt_font)],
            [g.Button('Detect Hand', button_color = bt_color, size = bt_size, font = bt_font)],
            [g.Button('Take Picture', button_color = bt_color, size = bt_size, font = bt_font)],
            [g.Button('Predict Visual', button_color = bt_color, size = bt_size, font = bt_font)]]


w , h = g.Window.get_screen_size()


window = g.Window("Hand Recognition Demo", layout, size = (w // 2, h // 2))


while True:
    event, values = window.read()
    if event == "Ex1" or event == g.WIN_CLOSED:
        break
window.close()