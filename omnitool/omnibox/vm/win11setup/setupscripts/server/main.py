import os
import logging
import argparse
import shlex
import subprocess
from flask import Flask, request, jsonify, send_file
import threading
import traceback
import pyautogui
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", help="log file path", type=str,
                    default=os.path.join(os.path.dirname(__file__), "server.log"))
parser.add_argument("--port", help="port", type=int, default=5000)
args = parser.parse_args()

logging.basicConfig(filename=args.log_file,level=logging.DEBUG, filemode='w' )
logger = logging.getLogger('werkzeug')

app = Flask(__name__)

computer_control_lock = threading.Lock()

@app.route('/probe', methods=['GET'])
def probe_endpoint():
    return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

@app.route('/execute', methods=['POST'])
def execute_command():
    # Only execute one command at a time
    with computer_control_lock:
        data = request.json
        # The 'command' key in the JSON request should contain the command to be executed.
        ALLOWLIST = {
            "ls": ["-l", "-a", "-h"],
            "echo": [],
            "cat": []
        }

        command = data.get('command', [])
        if not isinstance(command, list) or len(command) == 0:
            return jsonify({'status': 'error', 'message': 'Invalid command format'}), 400

        # Validate command against allowlist
        executable = command[0]
        args = command[1:]
        if executable not in ALLOWLIST or any(arg not in ALLOWLIST[executable] for arg in args):
            return jsonify({'status': 'error', 'message': 'Command not allowed'}), 403

        # Expand user directory
        for i, arg in enumerate(command):
            if arg.startswith("~/"):
                command[i] = os.path.expanduser(arg)

        # Execute the validated command
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, text=True, timeout=120)
            return jsonify({
                'status': 'success',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode
            })
        except Exception as e:
            logger.error("\n" + traceback.format_exc() + "\n")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():    
    cursor_path = os.path.join(os.path.dirname(__file__), "cursor.png")
    screenshot = pyautogui.screenshot()
    cursor_x, cursor_y = pyautogui.position()
    cursor = Image.open(cursor_path)
    # make the cursor smaller
    cursor = cursor.resize((int(cursor.width / 1.5), int(cursor.height / 1.5)))
    screenshot.paste(cursor, (cursor_x, cursor_y), cursor)
    

    # Convert PIL Image to bytes and send
    img_io = BytesIO()
    screenshot.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=args.port)
