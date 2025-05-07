import logging
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from instance import IInstance
from threading import Thread
from logging_utils import default_logger


def create_app(log_file=None):
    """Create and configure the Flask app"""
    if log_file:
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')
    logger = logging.getLogger('werkzeug')

    app = Flask(__name__)
    command_history = []

    @app.route('/probe', methods=['GET'])
    def probe_endpoint():
        return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

    @app.route('/execute', methods=['POST'])
    def execute_command():
        data = request.json
        command = data.get('command', "")
        command_history.append(str(command))
        return jsonify({
            'status': 'success',
            'output': '{}',
            'error': '',
            'returncode': 0
        })

    @app.route('/screenshot', methods=['GET'])
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
        max_commands = min(len(command_history), 20)  # Limit to last 20 commands
        
        if max_commands == 0:
            draw.text((20, y_position), "No commands executed yet", fill="gray", font=font)
        else:
            for i, cmd in enumerate(command_history[-max_commands:]):
                timestamp = f"{i+1}."
                draw.text((20, y_position), timestamp, fill="blue", font=font)
                draw.text((50, y_position), cmd, fill="black", font=font)
                y_position += 25
                
                # If we're running out of space
                if y_position > height - 30:
                    remaining = len(command_history) - (i + 1)
                    if remaining > 0:
                        draw.text((20, y_position), f"... {remaining} more commands not shown", 
                                 fill="red", font=font)
                    break
        
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    
    return app


class MockInstance(IInstance):
    def __init__(self, root_path = None, instance_num = 0, logger = None):
        print(f'Creating mock instance: {instance_num}')
        self.root_path = root_path
        self.instance_num = instance_num
        self.app = create_app()
        self.logger = logger or default_logger()
        p = Thread(target=self.app.run, args=('0.0.0.0',5000 + self.instance_num), daemon=True)
        p.start()

    def create(self):
        self.logger.info("Dummy instance created")

    def start(self):
        self.logger.info("Dummy instance started")

    def stop(self):
        self.logger.info("Dummy instance stopped")

    def delete(self):
        self.logger.info("Dummy instance deleted")

    def flask_url(self):
        return f"http://localhost:{5000 + self.instance_num}"

    def is_ready(self):
        return True
    
    def reset(self):
        self.logger.info("Dummy instance reset")

    def reset_soft(self):
        self.logger.info("Dummy instance reset soft")
