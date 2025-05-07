from fastapi import FastAPI, HTTPException
import requests
import uvicorn
from typing import Dict, Any
from contextlib import asynccontextmanager
import argparse
from instance_manager import InstanceManager
from instance_client import InstanceClient
from pathlib import Path

parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument('--path', type=str, default='/home/azureuser/ataymano/OmniParser/omnitool/omnibox/run1', help="Path to the instance directory. Expected to contain prepared common subfolder")
args = parser.parse_args()

Path(args.path).mkdir(parents=True, exist_ok=True)
instance_manager = InstanceManager(args.path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code goes here
    yield
    instance_manager.shutdown()


app = FastAPI(lifespan=lifespan)

@app.post("/get")
def get_instance():
    """Get an available instance from the pool"""
    instance_uuid = instance_manager.start()
    if not instance_uuid:
        raise HTTPException(status_code=503, detail="No instances available")
    return {"instance_uuid": instance_uuid}

@app.post("/reset")
def reset_instance(instance_id: str):
    """Reset an instance to its initial state and make it available again"""
    if instance_manager.reset(instance_id):
        return {"status": "success", "message": f"UUID {instance_id} for instance has been queued for reset"}
    raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_id}")

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

@app.get("/info")
def get_available_instances():
    """Get an available instance from the pool"""
    return {
        'available': len(instance_manager.available_instances),
        'capacity': instance_manager.capacity,
        'in_use': list(instance_manager.in_use.keys())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)

