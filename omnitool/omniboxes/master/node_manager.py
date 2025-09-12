import asyncio
import httpx
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging_utils


class NodeRegistration(BaseModel):
    url: str


class NodeStatus(BaseModel):
    url: str
    hash: str
    healthy: bool
    capacity: int
    available: int
    instances: List[str]

    @staticmethod
    def failed(url):
        return NodeStatus(url=url, hash=url_hash(url), healthy=False, capacity=0, available=0, instances=[])


def url_hash(url: str) -> str:
    return url


def _default_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("omnibox-master")


class NodeManager:
    def __init__(self, logger = None, update_timeout: int = 10):
        self._nodes = {}
        self._node_info = {}
        self.logger = logger or _default_logger()
        self.update_timeout = update_timeout

    async def _get_status(self, node_url):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{node_url}/info")
                if response.status_code != 200:
                    return NodeStatus.failed(node_url)
                data = response.json()
                return NodeStatus(
                    url = node_url,
                    hash = url_hash(node_url),
                    healthy = True,
                    capacity = data.get("capacity", 0),
                    available = data.get("available", 0),
                    instances = data.get("in_use", [])
                )
        except Exception as e:
            self.logger.warning(f"Node {node_url} health check failed: {str(e)}")
            return NodeStatus.failed(node_url)

    async def update_statuses(self):
        for node_url in self._nodes.keys():
            self._node_info[node_url] = await self._get_status(node_url)

    async def update_statuses_worker(self):
        while True:
            await self.update_statuses()
            await asyncio.sleep(self.update_timeout)

    async def register_node(self, node: NodeRegistration):
        self._nodes[url_hash(node.url)] = node
        self._node_info[url_hash(node.url)] = await self._get_status(node.url)            

    async def unregister_node(self, node_url: str):
        """Unregister a worker node from the master"""
        node_hash = url_hash(node_url)
        if node_hash not in self._nodes:
            self.logger.warning(f"Node {node_url} not found for unregistration")
            return False
        
        self._nodes.pop(node_hash, None)
        self._node_info.pop(node_hash, None)
        return True
        
    def get_best_node(self) -> Optional[str]:
        result = None
        available = 0
        for node in self._node_info.values():
            if node.healthy and node.available > available:
                result = node
                available = node.available
        return result
    
    def get_node(self, hash: str) -> Optional[NodeStatus]:
        return self._node_info.get(hash, None)

    def node_info(self):
        return self._node_info
    
