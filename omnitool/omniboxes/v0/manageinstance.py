import os
import subprocess
import tempfile
import threading
import queue
import shutil
import time
import requests
import concurrent.futures

# Base compose file template
compose_template = """
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
omniboxes_path = os.path.dirname(__file__)
def get_compose_path(instance_num):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        compose_content = compose_template.format(
            instance = instance_num,
            web_port = 8006 + instance_num,
            control_port = 5000 + instance_num,
            omniboxes_path = omniboxes_path
        )
        temp_file.write(compose_content)
        return temp_file.name

def create_instance(instance_num):
    if not os.path.exists(f"omnibox-{instance_num}"):
        os.makedirs(f"omnibox-{instance_num}")
        subprocess.run(["cp", "-r", f"{omniboxes_path}/common/win11storage/.", f"{omniboxes_path}/omnibox-{instance_num}/"])

    temp_file_path = get_compose_path(instance_num)
    try:
        subprocess.run(
            ["docker", "compose", "-f", temp_file_path, "-p", f"omnibox-{instance_num}", "up", "-d"],
            check=True,
            stdout=subprocess.DEVNULL,  # Suppress standard output
            stderr=subprocess.DEVNULL   # Suppress error output
        )
        print(f"Instance {instance_num} launched successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error launching instance {instance_num}: {e}")
    finally:
        os.unlink(temp_file_path) # Clean up the temporary file

def start_instance(instance_num):
    temp_file_path = get_compose_path(instance_num)
    try:
        subprocess.run(
            ["docker", "compose", "-f", temp_file_path, "-p", f"omnibox-{instance_num}", "start"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"Instance {instance_num} started successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error starting instance {instance_num}: {e}")
    finally:
        os.unlink(temp_file_path)

def stop_instance(instance_num):
    temp_file_path = get_compose_path(instance_num)
    try:
        subprocess.run(
            ["docker", "compose", "-f", temp_file_path, "-p", f"omnibox-{instance_num}", "stop"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"Instance {instance_num} stopped successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping instance {instance_num}: {e}")
    finally:
        os.unlink(temp_file_path)

def delete_instance(instance_num):
    temp_file_path = get_compose_path(instance_num)
    try:
        subprocess.run(
            ["docker", "compose", "-f", temp_file_path, "-p", f"omnibox-{instance_num}", "down"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"Instance {instance_num} deleted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error deleting instance {instance_num}: {e}")
    finally:
        os.unlink(temp_file_path)

def flask_url(instance_id: int):
    return f"http://localhost:{5000 + instance_id}"

def instance_ready(instance_id: int):
    try:
        response = requests.get(flask_url(instance_id) + "/probe", timeout=1)
        return response.status_code == 200
    except requests.RequestException as e:
        return False

def reset_instance(instance_id):
    delete_instance(instance_id)
    if os.path.exists(f"omnibox-{instance_id}"):
        subprocess.run(["sudo", "rm", "-rf", f"{omniboxes_path}/omnibox-{instance_id}"], check=True)
    create_instance(instance_id)

def reset_instance_soft(instance_id):
    stop_instance(instance_id)
    if os.path.exists(f"omnibox-{instance_id}"):
        subprocess.run(["sudo", "rm", "-rf", f"{omniboxes_path}/omnibox-{instance_id}"], check=True)
    start_instance(instance_id)
    
def reset_instance_with_callback(instance_id, callback):
    try:
        reset_instance(instance_id)
        while not instance_ready(instance_id):
            time.sleep(1)
        callback(instance_id)
    except Exception as e:
        print(f"Error in reset worker: {str(e)}")

# got insufficient perf with this method
def reset_instance_soft_with_callback(instance_id, callback):
    try:
        reset_instance_soft(instance_id)
        while not instance_ready(instance_id):
            time.sleep(1)
        callback(instance_id)
    except Exception as e:
        print(f"Error in reset worker: {str(e)}")

# num_instances = 3
# for i in range(num_instances):
    # create_instance(i)
    # stop_instance(i)
    # start_instance(i)
    # delete_instance(i)
