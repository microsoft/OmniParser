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

def _position(ignore_by_mock = True):
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'p = pyautogui.position()',
            "print(json.dumps({'x': p.x, 'y': p.y}))",       
        ]),
        'ignore_by_mock': ignore_by_mock
    }

def _screensize(ignore_by_mock = True):
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'sz = pyautogui.size()',
            "print(json.dumps({'width': sz.width, 'height': sz.height}))",       
        ]),
        'ignore_by_mock': ignore_by_mock
    }


class InstanceClient:
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
        

class NodeClient:
    def __init__(self, host = 'localhost', port = 8000):
        self.host = host
        self.port = port
        self.output = None
        self.instance_id = None

    def get_instance(self):
        data = requests.post(f'http://{self.host}:{self.port}/get').json()
        return data.get('instance_id', None)
    
    def get_instances_info(self):
        data = requests.get(f'http://{self.host}:{self.port}/info')
        return data.json()    
    
    def reset_instance(self, instance_id):
        data = requests.post(f'http://{self.host}:{self.port}/reset', params = {'instance_id': instance_id})
        if data.status_code != 200:
            raise Exception(f"Error: {data.status_code} - {data.text}")
        return data.json()

    def screenshot(self, instance_id):
        data = requests.get(f'http://{self.host}:{self.port}/screenshot', params = {'instance_id': instance_id})
        image_data = io.BytesIO(data.content)
        return Image.open(image_data)

    def execute(self, instance_id, command):
        return requests.post(
            f'http://{self.host}:{self.port}/execute',
            params={'instance_id': instance_id},
            json = command)
    
    def do_and_show(self, instance_id, command, waitTime):
        response = self.execute(instance_id, command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        time.sleep(waitTime)
        self._display(instance_id, response.json()['output'])
    
    def _display(self, instance_id, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot(instance_id))
            print(data)

    def position(self, instance_id):
        return json.loads(self.execute(instance_id, _position()).json()['output'])
    
    def screensize(self, instance_id):
        return json.loads(self.execute(instance_id, _screensize()).json()['output'])    

    def ui(self):
        self.output = ipywidgets.Output()

        get_button = ipywidgets.Button(description = 'Get Instance')
        release_button = ipywidgets.Button(description = 'Release Instance')
        instances = ipywidgets.RadioButtons(description = 'Instances', options = self.get_instances_info()['in_use'], layout=ipywidgets.Layout(width='50%'))
        self.instance_id = instances.value
        def on_add_click(b):
            instance_id = self.get_instance()
            if instance_id:
                instances.options = list(instances.options) + [instance_id]
            else:
                print('No instance available')

        def on_release_click(b):
            self.reset_instance(instances.value)
            options = list(instances.options)
            options.remove(instances.value)
            instances.options = options       

        get_button.on_click(on_add_click)
        release_button.on_click(on_release_click)        

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        screensize = {}
        position = {}
        if self.instance_id:
            screensize = self.screensize(self.instance_id)
            position = self.position(self.instance_id)
            self._display(self.instance_id)

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(self.instance_id, _moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(self.instance_id, _moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(self.instance_id, _click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(self.instance_id, _rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(self.instance_id, _doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = True)
        submit = ipywidgets.Button(description = 'Submit')
        submit.on_click(lambda v: self.do_and_show(self.instance_id, {
            'command': _pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
            'shell': shell.value
        }, waitTime.value))

        def on_instance_change(_):
            self.instance_id = instances.value
            screensize = self.screensize(self.instance_id)
            x.max = screensize.get('width', 1)
            y.max = screensize.get('height', 1)
            position = self.position(self.instance_id)
            x.value = position.get('x', 0)
            y.value = position.get('y', 0)
            self._display(self.instance_id)

        instances.observe(on_instance_change, names='value')

        display(ipywidgets.VBox([
            ipywidgets.HBox([instances, ipywidgets.VBox([get_button, release_button])]),
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python]), submit]),
            self.output]))
        

class MasterClient:
    def __init__(self, host = 'localhost', port = 7000):
        self.host = host
        self.port = port
        self.output = None
        self.instance = None

    def probe(self, instance: dict):
        return requests.get(f'http://{self.host}:{self.port}/probe', params=instance).json()

    def get_instance(self):
        url = f'http://{self.host}:{self.port}/get'
        response = requests.post(url)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response.json()
    
    def get_info(self):
        url = f'http://{self.host}:{self.port}/info'
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        return response.json()
    
    def get_options(self):
        info = self.get_info()['nodes']
        print(type(info))
        return [json.dumps({'instance_id': instance, 'node': node['hash']}) for node in info for instance in node['instances']]

    def reset_instance(self, instance: dict):
        data = requests.post(f'http://{self.host}:{self.port}/reset', params = instance)
        if data.status_code != 200:
            raise Exception(f"Error: {data.status_code} - {data.text}")
        return data.json()

    def screenshot(self, instance: dict):
        data = requests.get(f'http://{self.host}:{self.port}/screenshot', params = instance)
        image_data = io.BytesIO(data.content)
        return Image.open(image_data)

    def execute(self, instance: dict, command):
        return requests.post(f'http://{self.host}:{self.port}/execute', json = dict(command, **instance))
    
    def do_and_show(self, instance, command, waitTime):
        response = self.execute(instance, command)
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        time.sleep(waitTime)
        self._display(instance, response.json()['output'])
    
    def _display(self, instance, data = None):
        with self.output:
            self.output.clear_output(wait = True)
            display(self.screenshot(instance))
            print(data)

    def position(self, instance):
        return json.loads(self.execute(instance, _position()).json()['output'])
    
    def screensize(self, instance):
        return json.loads(self.execute(instance, _screensize()).json()['output'])    

    def ui(self):
        self.output = ipywidgets.Output()

        get_button = ipywidgets.Button(description = 'Get Instance')
        release_button = ipywidgets.Button(description = 'Release Instance')
        instances = ipywidgets.RadioButtons(description = 'Instances', options = self.get_options(), layout=ipywidgets.Layout(width='50%'))
        self.instance = json.loads(instances.value) if instances.value else {}
        def on_add_click(b):
            new_instance = json.dumps(self.get_instance())
            instances.options = list(instances.options) + [new_instance]

        def on_release_click(b):
            self.reset_instance(json.loads(instances.value))
            options = list(instances.options)
            options.remove(instances.value)
            instances.options = options        

        get_button.on_click(on_add_click)
        release_button.on_click(on_release_click)        

        waitTime = ipywidgets.FloatLogSlider(base=2, value = 1, min = -3, max = 5, step = 1, description = 'wait (s)')

        screensize = {}
        position = {}
        if self.instance:
            screensize = self.screensize(self.instance)
            position = self.position(self.instance)
            self._display(self.instance)

        x = ipywidgets.IntSlider(min = 0, max = screensize.get('width', 1), value = position.get('x', 0), description = 'X')
        y = ipywidgets.IntSlider(min = 0, max = screensize.get('height', 1), value = position.get('y', 0), description = 'Y')

        click = ipywidgets.Button(description = 'Click')
        rightClick = ipywidgets.Button(description = 'Right Click')
        doubleClick = ipywidgets.Button(description = 'Double Click')

        x.observe(lambda v: self.do_and_show(self.instance, _moveTo(x.value, y.value), waitTime.value), names='value')
        y.observe(lambda v: self.do_and_show(self.instance, _moveTo(x.value, y.value), waitTime.value), names='value')

        click.on_click(lambda v: self.do_and_show(self.instance, _click(), waitTime.value))
        rightClick.on_click(lambda v: self.do_and_show(self.instance, _rightClick(), waitTime.value))
        doubleClick.on_click(lambda v: self.do_and_show(self.instance, _doubleClick(), waitTime.value))

        cmd = ipywidgets.Textarea(description = 'Commands', layout=ipywidgets.Layout(width='50%'))
        shell = ipywidgets.Checkbox(description = 'Shell', value = False)
        python = ipywidgets.Checkbox(description = 'Python', value = True)
        submit = ipywidgets.Button(description = 'Submit')
        submit.on_click(lambda v: self.do_and_show(self.instance, {
            'command': _pyscript(cmd.value.split('\n')) if python.value else '&&'.join(cmd.value.split('\n')),
            'shell': shell.value
        }, waitTime.value))

        def on_instance_change(_):
            self.instance = json.loads(instances.value)
            screensize = self.screensize(self.instance)
            x.max = screensize.get('width', 1)
            y.max = screensize.get('height', 1)
            position = self.position(self.instance)
            x.value = position.get('x', 0)
            y.value = position.get('y', 0)
            self._display(self.instance)

        instances.observe(on_instance_change, names='value')

        display(ipywidgets.VBox([
            ipywidgets.HBox([instances, ipywidgets.VBox([get_button, release_button])]),
            waitTime,
            ipywidgets.HBox([x, y]),
            ipywidgets.HBox([click, rightClick, doubleClick]),
            ipywidgets.HBox([cmd, ipywidgets.VBox([shell, python]), submit]),
            self.output]))

