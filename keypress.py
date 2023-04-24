import keyboard
import sys

def on_key_press(event):
    print('Key {} was pressed'.format(event.name))
    if event.name=='space':
        #keyboard.unhook_all()
        sys.exit()

keyboard.on_press(on_key_press) 

keyboard.wait()
