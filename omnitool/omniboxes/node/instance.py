import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any
from logging_utils import default_logger


class IInstance:
    def create(self):
        raise NotImplementedError("Subclasses should implement this!")

    def start(self):
        raise NotImplementedError("Subclasses should implement this!")

    def stop(self):
        raise NotImplementedError("Subclasses should implement this!")

    def delete(self):
        raise NotImplementedError("Subclasses should implement this!")

    def flask_url(self):
        raise NotImplementedError("Subclasses should implement this!")

    def is_ready(self):
        raise NotImplementedError("Subclasses should implement this!")

    def reset(self):
        raise NotImplementedError("Subclasses should implement this!")

    def reset_soft(self):
        raise NotImplementedError("Subclasses should implement this!")


_compose_template = """
networks:
  omnibox-network:
    name: omnibox-network-{instance}

services:
  windows:
    image: windows-local
    container_name: omni-windows-{instance}
    networks:
      - omnibox-network
    privileged: true
    environment:
      RAM_SIZE: "2G"
      CPU_CORES: "4"
      DISK_SIZE: "11G"
    devices:
      - /dev/kvm
      - /dev/net/tun
    cap_add:
      - NET_ADMIN
    ports:
      - {web_port}:8006       # Web Viewer access
      - {control_port}:5000   # Computer control server
    volumes:
      - {omniboxes_path}/common/win11iso/custom.iso:/custom.iso
      - {omniboxes_path}/common/win11setup/firstboot:/oem
      - {omniboxes_path}/common/win11setup/setupscripts:/data
      - {omniboxes_path}/omnibox-{instance}:/storage
"""


class Instance(IInstance):
    def __init__(self, root_path = None, instance_num = 0, logger = None):
        self.root_path = Path(root_path or os.path.dirname(__file__)).resolve()
        self.instance_num = instance_num
        self.config_path = self.root_path / f'{instance_num}.yml'
        self.logger = logger or default_logger()
        with open(self.config_path, mode='w') as temp_file:
            compose_content = _compose_template.format(
                instance = self.instance_num,
                web_port = 8006 + self.instance_num,
                control_port = 5000 + self.instance_num,
                omniboxes_path = self.root_path
            )
            temp_file.write(compose_content)

    def path(self):
        return f"{self.root_path}/omnibox-{self.instance_num}/"

    def _execute(self, command):
        self.logger.info(f'Running: {" ".join(command)}')
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def create(self):
        if not os.path.exists(self.path()):
            os.makedirs(f"omnibox-{self.instance_num}")
            subprocess.run(["cp", "-r", f"{str(self.root_path)}/common/win11storage/.", self.path()])

        try:
            self._execute(["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "up", "-d"])
            self.logger.info(f"Instance {self.instance_num} launched successfully!")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error launching instance {self.instance_num}: {e}")

    def start(self):
        try:
            self._execute(["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "start"])
            self.logger.info(f"Instance {self.instance_num} started successfully!")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error starting instance {self.instance_num}: {e}")

    def stop(self):
        try:
            self._execute(["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "stop"])
            self.logger.info(f"Instance {self.instance_num} stopped successfully!")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error stopping instance {self.instance_num}: {e}")

    def delete(self):
        try:
            self._execute(["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "down"])
            self.logger.info(f"Instance {self.instance_num} deleted successfully!")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error deleting instance {self.instance_num}: {e}")

    def flask_url(self):
        return f"http://localhost:{5000 + self.instance_num}"

    def is_ready(self):
        try:
            response = requests.get(self.flask_url() + "/probe", timeout=1)
            return response.status_code == 200
        except requests.RequestException as e:
            return False

    def reset(self):
        self.delete()
        if os.path.exists(self.path()):
            subprocess.run(["sudo", "rm", "-rf", f"{self.root_path}/omnibox-{self.instance_num}"], check=True)
        self.create()

    def reset_soft(self):
        self.stop()
        if os.path.exists(self.path()):
            subprocess.run(["sudo", "rm", "-rf", f"{self.root_path}/omnibox-{self.instance_num}"], check=True)
        self.start()


def reset_with_callback(instance, callback):
    try:
        instance.reset()
        while not instance.is_ready():
            time.sleep(1)
        callback(instance)
    except Exception as e:
        instance.logger.error(f"Error in reset worker: {str(e)}")


# got insufficient perf with this method
def reset_soft_with_callback(instance, callback):
    try:
        instance.reset_soft()
        while not instance.is_ready():
            time.sleep(1)
        callback(instance)
    except Exception as e:
        instance.logger.error(f"Error in reset worker: {str(e)}")

