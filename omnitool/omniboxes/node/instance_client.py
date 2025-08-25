from typing import Dict, Any
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Response
import requests

class InstanceClient:
    def __init__(self, url):
        self.url = url

    def execute(self, command_data: Dict[str, Any]):
        """Forward execute command to the Flask server in the specified instance"""
        try:
            response = requests.post(f"{self.url}/execute", json=command_data, timeout=5)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with theinstance {self.url}: {str(e)}")

    def screenshot(self):
        """Forward screenshot request to the Flask server in the specified instance"""
        try:
            response = requests.get(f"{self.url}/screenshot", timeout=5)
            
            if response.status_code == 200:
                return Response(content=response.content, media_type="image/png")
            else:
                return JSONResponse(
                    content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
                    status_code=response.status_code
                )
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the instance {self.url}: {str(e)}")

    def probe(self):
        """Forward probe request to the Flask server in the specified instance"""
        try:
            response = requests.get(f"{self.url}/probe", timeout=1)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the instance {self.url}: {str(e)}")
