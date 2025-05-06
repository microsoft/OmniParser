import os
import subprocess
import tempfile
import time
import requests
from pathlib import Path


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
        self.stop()
        if os.path.exists(f"omnibox-{self.instance_num}"):
            subprocess.run(["sudo", "rm", "-rf", f"{self.root_path}/omnibox-{self.instance_num}"], check=True)
        self.start()


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
    def __init__(self, root_path = None, instance_num = 0):
        self.root_path = Path(root_path or os.path.dirname(__file__))
        self.instance_num = instance_num
        self.config_path = self.root_path / f'{instance_num}.yml'
        with open(self.config_path, mode='w') as temp_file:
            compose_content = _compose_template.format(
                instance = self.instance_num,
                web_port = 8006 + self.instance_num,
                control_port = 5000 + self.instance_num,
                omniboxes_path = self.root_path
            )
            temp_file.write(compose_content)

    def create(self):
        if not os.path.exists(f"omnibox-{self.instance_num}"):
            os.makedirs(f"omnibox-{self.instance_num}")
            subprocess.run(["cp", "-r", f"{str(self.root_path)}/common/win11storage/.", f"{self.root_path}/omnibox-{self.instance_num}/"])

        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "up", "-d"],
                check=True,
                stdout=subprocess.DEVNULL,  # Suppress standard output
                stderr=subprocess.DEVNULL   # Suppress error output
            )
            print(f"Instance {self.instance_num} launched successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error launching instance {self.instance_num}: {e}")

    def start(self):
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "start"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Instance {self.instance_num} started successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error starting instance {self.instance_num}: {e}")

    def stop(self):
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "stop"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Instance {self.instance_num} stopped successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error stopping instance {self.instance_num}: {e}")

    def delete(self):
        try:
            subprocess.run(
                ["docker", "compose", "-f", str(self.config_path), "-p", f"omnibox-{self.instance_num}", "down"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Instance {self.instance_num} deleted successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting instance {self.instance_num}: {e}")

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
        if os.path.exists(f"omnibox-{self.instance_num}"):
            subprocess.run(["sudo", "rm", "-rf", f"{self.root_path}/omnibox-{self.instance_num}"], check=True)
        self.create()

    def reset_soft(self):
        self.stop()
        if os.path.exists(f"omnibox-{self.instance_num}"):
            subprocess.run(["sudo", "rm", "-rf", f"{self.root_path}/omnibox-{self.instance_num}"], check=True)
        self.start()


def reset_with_callback(instance, callback):
    try:
        instance.reset()
        while not instance.is_ready():
            time.sleep(1)
        callback(instance)
    except Exception as e:
        print(f"Error in reset worker: {str(e)}")


# got insufficient perf with this method
def reset_soft_with_callback(instance, callback):
    try:
        instance.reset_soft()
        while not instance.is_ready():
            time.sleep(1)
        callback(instance)
    except Exception as e:
        print(f"Error in reset worker: {str(e)}")

