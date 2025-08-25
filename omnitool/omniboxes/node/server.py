from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from typing import Dict, Any
from contextlib import asynccontextmanager
import argparse
from instance_manager import InstanceManager
from instance_client import InstanceClient
from instance import Instance
from mock_instance import MockInstance
from pathlib import Path

parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument('--path', type=str, default='../run', help="Path to the instance directory. Expected to contain prepared common subfolder")
parser.add_argument('--base_control_port', type=int, default=5000, help="Base control port offset for the instances (for testing with mock instances)")
parser.add_argument('--mock', action='store_true', help="Use mock instances")
args = parser.parse_args()

Path(args.path).mkdir(parents=True, exist_ok=True)
mock_instance_factory = lambda root_path, instance_num, logger: MockInstance(root_path, instance_num, logger, args.base_control_port)
instance_factory = mock_instance_factory if args.mock else Instance
instance_manager = InstanceManager(instance_factory = instance_factory, path = args.path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code goes here
    yield
    instance_manager.shutdown()


app = FastAPI(lifespan=lifespan)

@app.post("/get")
def get_instance():
    """Get an available instance from the pool"""
    instance_id = instance_manager.start()
    if not instance_id:
        raise HTTPException(status_code=503, detail="No instances available")
    return {"instance_id": instance_id}

@app.post("/reset")
def reset_instance(instance_id: str):
    """Reset an instance to its initial state and make it available again"""
    if instance_manager.reset(instance_id):
        return {"status": "success", "message": f"UUID {instance_id} for instance has been queued for reset"}
    raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")

@app.get("/probe")
async def probe_instance(instance_id: str):
    """Forward probe request to the Flask server in the specified instance"""
    instance= instance_manager.get(instance_id)
    if not instance:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")
    try:
        return InstanceClient(instance.flask_url()).probe()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with UUID {instance_id} for instance {instance.instance_num}: {str(e)}")

@app.get("/screenshot")
async def get_instance_screenshot(instance_id: str):
    """Forward screenshot request to the Flask server in the specified instance"""
    instance = instance_manager.get(instance_id)
    if not instance:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")
    try:
        return InstanceClient(instance.flask_url()).screenshot()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with UUID {instance_id} for instance {instance.instance_num}: {str(e)}")

@app.post("/execute")
async def execute_instance_command(instance_id: str, command_data: Dict[str, Any]):
    """Forward execute command to the Flask server in the specified instance"""
    instance= instance_manager.get(instance_id)
    if not instance:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")
    try:
        return InstanceClient(instance.flask_url()).execute(command_data)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with instance {instance.instance_num}: {str(e)}")

@app.get("/info")
def get_available_instances():
    """Get an available instance from the pool"""
    return {
        'available': len(instance_manager.available_instances),
        'capacity': instance_manager.capacity,
        'in_use': list(instance_manager.in_use.keys())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)

