import vgamepad as vg
import time

gamepad = vg.VDS4Gamepad()

while True:
    gamepad.left_joystick(x_value=32768, y_value=0)
    gamepad.update()
    time.sleep(0.5)
    gamepad.left_joystick(x_value=-32768, y_value=0)
    gamepad.update()
    time.sleep(0.5)

input()