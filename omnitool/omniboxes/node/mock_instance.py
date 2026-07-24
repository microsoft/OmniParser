import logging
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from instance import IInstance
from threading import Thread
from logging_utils import default_logger


class MockApp:
    def __init__(self, log_file=None):
        """Create and configure the Flask app"""
        if log_file:
            logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')
        logger = logging.getLogger('werkzeug')

        self.app = Flask(__name__)
        self.command_history = []

        @self.app.route('/probe', methods=['GET'])
        def probe_endpoint():
            return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

        @self.app.route('/execute', methods=['POST'])
        def execute_command():
            data = request.json
            command = data.get('command', "")
            if not data.get('ignore_by_mock', False):
                self.command_history.append(str(command))
            return jsonify({
                'status': 'success',
                'output': '{}',
                'error': '',
                'returncode': 0
            })

        @self.app.route('/screenshot', methods=['GET'])
        def screenshot():
            width, height = 800, 600
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
                
            draw.text((10, 10), "Mock Instance - Command History", fill="black", font=font)
            draw.line([(10, 40), (width-10, 40)], fill="black", width=1)
            
            y_position = 50
            max_commands = min(len(self.command_history), 20)  # Limit to last 20 commands
            
            if max_commands == 0:
                draw.text((20, y_position), "No commands executed yet", fill="gray", font=font)
            else:
                for i, cmd in enumerate(self.command_history[-max_commands:]):
                    timestamp = f"{i+1}."
                    draw.text((20, y_position), timestamp, fill="blue", font=font)
                    draw.text((50, y_position), cmd, fill="black", font=font)
                    y_position += 25
                    
                    # If we're running out of space
                    if y_position > height - 30:
                        remaining = len(self.command_history) - (i + 1)
                        if remaining > 0:
                            draw.text((20, y_position), f"... {remaining} more commands not shown", 
                                    fill="red", font=font)
                        break
            
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        


class MockInstance(IInstance):
    def __init__(self, root_path = None, instance_num = 0, logger = None, base_control_port = 5000):
        print(f'Creating mock instance: {instance_num}')
        self.root_path = root_path
        self.instance_num = instance_num
        self.app = MockApp()
        self.logger = logger or default_logger()
        self.base_control_port = base_control_port
        p = Thread(target=self.app.app.run, args=('0.0.0.0', self.control_port), daemon=True)
        p.start()

    @property
    def control_port(self):
        return self.base_control_port + self.instance_num

    def create(self):
        self.logger.info(f"Creating dummy instance {self.instance_num}")

    def start(self):
        self.logger.info(f"Starting dummy instance {self.instance_num}")

    def stop(self):
        self.logger.info(f"Stopping dummy instance {self.instance_num}")

    def delete(self):
        self.logger.info(f"Deleting dummy instance {self.instance_num}")

    def flask_url(self):
        return f"http://localhost:{self.control_port}"

    def is_ready(self):
        return True
    
    def reset(self):
        self.app.command_history = []
        self.logger.info(f"Resetting dummy instance {self.instance_num}")

    def reset_soft(self):
        self.logger.info(f"Soft resetting dummy instance {self.instance_num}")
