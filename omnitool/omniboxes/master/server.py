import asyncio
import httpx
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from node_manager import NodeManager, NodeRegistration
import requests
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("omnibox-master")

node_manager = NodeManager(logger = logger)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await node_manager.register_node(NodeRegistration(url = "http://localhost:8000"))
#    await node_manager.register_node(NodeRegistration(url = "http://localhost:8001"))
    tasks = asyncio.create_task(node_manager.update_statuses_worker())
    yield
    # Shutdown code goes here

app = FastAPI(
    title="OmniBox Master Node",
    description="Manages redirection of instance operations to worker nodes",
    lifespan = lifespan)

@app.post("/get")
def create_instance():
    """Get an available new instance from less occupied node"""
    node = node_manager.get_best_node()
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No available nodes with capacity to create new instance"
        )
    
    data = requests.post(f'{node.url}/get').json()
    if 'instance_id' in data:
        return {
            'instance_id': data['instance_id'],
            'node': node.hash,
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No available nodes with capacity to create new instance"
    )

@app.post("/reset")
async def reset(instance_id: str, node: str):
    """Reset an existing instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.post(f'{node_info.url}/reset', params={"instance_id": instance_id})
    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/probe")
async def probe(instance_id: str, node: str):
    """Probe the instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.get(f'{node_info.url}/probe', params={"instance_id": instance_id})
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.get("/screenshot")
async def screenshot(instance_id: str, node: str):
    """Make a screenshot of an existing instance by delegating to the worker node that hosts it"""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.get(f'{node_info.url}/screenshot', params={"instance_id": instance_id})   
    if response.status_code == 200:
        return Response(content=response.content, media_type="image/png")

    return JSONResponse(
        content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
        status_code=response.status_code
    )

@app.post("/execute")
async def execute(command_data: Dict[str, Any]):
    """Forward execute command to the Flask server in the specified instance"""
    node = command_data.pop('node')
    instance_id = command_data.pop('instance_id')
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found"
        )
    response = requests.post(f'{node_info.url}/execute',
                            params={"instance_id": instance_id},
                            json = command_data)   
    if response.status_code == 200:
        return JSONResponse(content=response.json(), status_code=response.status_code)
    
    return JSONResponse(
        content={"status": "error", "message": f"Failed to execute command: {response.text}"},
        status_code=response.status_code
    )

@app.get("/info")
def get_info():
    node_info = node_manager.node_info()
    return JSONResponse(
        content={
            "nodes": [
                {
                    "url": node.url,
                    "hash": node.hash,
                    "healthy": node.healthy,
                    "capacity": node.capacity,
                    "available": node.available,
                    "instances": node.instances
                }
                for node in node_info.values()
            ]
        },
        status_code=status.HTTP_200_OK
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniBox Host")
    parser.add_argument("--port", type=int, default=7000, help="Port to run the server on")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)