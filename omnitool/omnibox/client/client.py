import requests
from PIL import Image
import io
import ipywidgets
import json
import time

def _pyscript(commands):
    script = ";".join(commands)
    return f'python -c "{script}"'

def _moveTo(x, y):
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.moveTo({x}, {y})'
        ])
    }

def _click():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.click()'
        ])
    }

def _rightClick():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.rightClick()'
        ])
    }

def _doubleClick():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.doubleClick()'
        ])
    }

def _position():
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'p = pyautogui.position()',
            "print(json.dumps({'x': p.x, 'y': p.y}))",       
        ])
    }

def _screensize():
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'sz = pyautogui.size()',
            "print(json.dumps({'width': sz.width, 'height': sz.height}))",       
        ])
    }


class Client:
    def __init__(self, host = 'localhost', port = 5000):
        self.host = host
        self.port = port
        self.output = None

    def screenshot(self):
        data = requests.get(f'http://{self.host}:{self.port}/screenshot')
        image_data = io.BytesIO(data.content)
        return Image.open(image_data)

    def execute(self, command):
        return requests.post(f'http://{self.host}:{self.port}/execute', json = command)
    
    def do_and_show(self, command, waitTime):
        response = self.execute(command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        time.sleep(waitTime)
        self._display(response.json()['output'])
    
    def _display(self, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot())
            print(data)

    def position(self):
        return json.loads(self.execute(_position()).json()['output'])
    
    def screensize(self):
        return json.loads(self.execute(_screensize()).json()['output'])    

    def ui(self):
        self.output = ipywidgets.Output()
        self._display()

        screensize = self.screensize()
        position = self.position()

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(_moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(_moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(_click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(_rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(_doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = True)
        submit = ipywidgets.Button(description = 'Submit')
        submit.on_click(lambda v: self.do_and_show({
            'command': _pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
            'shell': shell.value
        }, waitTime.value))
        display(ipywidgets.VBox([
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python]), submit]),
            self.output]))
