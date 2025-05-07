import concurrent
from instance import Instance, reset_with_callback
#from mock_instance import MockInstance as Instance
from tqdm import tqdm
import time
import uuid
from typing import Dict, Any
from pathlib import Path
import os


class InstanceManager:
    def __init__(self, path = None, capacity: int = 2):
        self.capacity = capacity
        self.available_instances = {} # instance_num to instance
        self.in_use = {} # key = instance_uuid, value = instance_id

        self.reset_workers = capacity
        self.reset_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.reset_workers)
        self.path = Path(path or os.path.dirname(__file__)).resolve()

        for i in range(self.capacity):
            self.reset_executor.submit(reset_with_callback, Instance(self.path, instance_num=i), self.instance_reset_callback)

        # Wait for all instances to be initialized
        with tqdm(total=self.capacity, desc="Initializing instances") as pbar:
            last_count = 0
            while len(self.available_instances) < self.capacity:
                current_count = len(self.available_instances)
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)
            pbar.update(self.capacity - last_count)

    def instance_reset_callback(self, instance):
        self.available_instances[instance.instance_num] = instance

    def shutdown(self):
        if self.reset_executor:
            self.reset_executor.shutdown(wait=True, cancel_futures=False)

    def start(self):
        if not self.available_instances:
            return None
        
        instance = self.available_instances.pop(list(self.available_instances.keys())[0])
        instance_uuid = str(uuid.uuid4())
        self.in_use[instance_uuid] = instance
        return instance_uuid
    
    def reset(self, instance_uuid: str):
        if instance_uuid not in self.in_use:
            return False
        
        instance = self.in_use[instance_uuid]
        self.reset_executor.submit(reset_with_callback, instance, self.instance_reset_callback)
        return True
    
    def get(self, uuid: str):
        return self.in_use.get(uuid, None)
    