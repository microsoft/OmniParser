import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import uvicorn
import concurrent.futures
from typing import List, Dict, Any, Optional
from manageinstance import (
 reset_instance_with_callback, flask_url
)
import time
import uuid
from tqdm import tqdm
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code goes here
    yield
    # Shutdown code goes here
    if reset_executor:
        reset_executor.shutdown(wait=True, cancel_futures=False)

app = FastAPI(lifespan=lifespan)

INSTANCES = 10
available_instances = []
in_use = {} # key = instance_uuid, value = instance_id

RESET_WORKERS = 10
reset_executor = concurrent.futures.ThreadPoolExecutor(max_workers=RESET_WORKERS)

def instance_reset_callback(instance_id):
    if instance_id not in available_instances:
        available_instances.append(instance_id)

for i in range(INSTANCES):
    reset_executor.submit(reset_instance_with_callback, i, instance_reset_callback)

# Wait for all instances to be initialized
with tqdm(total=INSTANCES, desc="Initializing instances") as pbar:
    last_count = 0
    while len(available_instances) < INSTANCES:
        current_count = len(available_instances)
        if current_count > last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
        time.sleep(0.1)
    pbar.update(INSTANCES - last_count)

@app.get("/getinstance")
def get_instance():
    """Get an available instance from the pool"""
    if not available_instances:
        raise HTTPException(status_code=503, detail="No instances available")
    
    instance_id = available_instances.pop(0)
    instance_uuid = str(uuid.uuid4())
    in_use[instance_uuid] = instance_id
    return {"instance_uuid": instance_uuid}

@app.post("/resetinstance/{instance_uuid}")
def reset_instance(instance_uuid: str):
    """Reset an instance to its initial state and make it available again"""
    if instance_uuid not in in_use:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_uuid}")
    
    instance_id = in_use[instance_uuid]
    reset_executor.submit(reset_instance_with_callback, instance_id, instance_reset_callback)
    
    return {"status": "success", "message": f"UUID {instance_uuid} for instance {instance_id} has been queued for reset"}

@app.post("/executeinstance/{instance_uuid}/execute")
async def execute_instance_command(instance_uuid: str, command_data: Dict[str, Any]):
    """Forward execute command to the Flask server in the specified instance"""
    if instance_uuid not in in_use:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_uuid}")
    
    instance_id = in_use[instance_uuid]
    try:
        response = requests.post(flask_url(instance_id), json=command_data, timeout=5)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with instance {instance_id}: {str(e)}")

@app.get("/executeinstance/{instance_uuid}/screenshot")
async def get_instance_screenshot(instance_uuid: str):
    """Forward screenshot request to the Flask server in the specified instance"""
    if instance_uuid not in in_use:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_uuid}")
    
    instance_id = in_use[instance_uuid]
    try:
        response = requests.get(flask_url(instance_id) + "/screenshot", timeout=5)
        
        if response.status_code == 200:
            return Response(content=response.content, media_type="image/png")
        else:
            return JSONResponse(
                content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
                status_code=response.status_code
            )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with UUID {instance_uuid} for instance {instance_id}: {str(e)}")

@app.get("/executeinstance/{instance_uuid}/probe")
async def probe_instance(instance_uuid: str):
    """Forward probe request to the Flask server in the specified instance"""
    if instance_uuid not in in_use:
        raise HTTPException(status_code=400, detail=f"Invalid instance UUID: {instance_uuid}")
    
    instance_id = in_use[instance_uuid]
    try:
        response = requests.get(flask_url(instance_id) + "/probe", timeout=1)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with UUID {instance_uuid} for instance {instance_id}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

