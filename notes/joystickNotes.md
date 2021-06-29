# Gamepad Emulator Notes

Using vGamepad kita bisa mem-bind controller di euro truck simulator.

Dari Library vGamepad contoh:
```py
gamepad.left_trigger(value=100)  # value between 0 and 255
gamepad.right_trigger(value=255)  # value between 0 and 255
gamepad.left_joystick(x_value=-10000, y_value=0)  # values between -32768 and 32767
gamepad.right_joystick(x_value=-32768, y_value=15000)  # values between -32768 and 32767
gamepad.update() # Selalu digunakan setelah set value
```
setiap ngeset value, kita harus melakukan gamepad update

<hr>

## Setting up the keybind

1.  `gamepad.left_trigger(value=255)` for brakes with 0 is no brake and 255 is brake
2.  `gamepad.right_trigger(value=255)` for accelerate with 0 is no brake and 255 is accelerate
3. `gamepad.left_joystick(x_value=-10000, y_value=0)` for steering right and left
