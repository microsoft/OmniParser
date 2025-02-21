import modal
import sys
import logging
from pathlib import Path
import re
from typing import List, Dict, Optional
import subprocess
import tempfile
import json
import os
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_NAME = os.environ.get("MODAL_APP_NAME", "omniparser-v2")

@dataclass
class DeploymentResult:
    """Represents the result of a deployment."""
    instance_number: int
    url: Optional[str]
    success: bool
    error: Optional[str]
    deploy_time: float
    timestamp: datetime

class ModalDeployer:
    def __init__(self, base_app_name: str = APP_NAME):
        self.base_app_name = base_app_name
        self.base_app_path = Path(__file__).parent / "modal_app.py"
        self.deployments: Dict[int, DeploymentResult] = {}
        
        if not self.base_app_path.exists():
            raise FileNotFoundError(f"Base app file not found: {self.base_app_path}")

    def _deploy_single(self, instance_number: int) -> DeploymentResult:
        """Deploy a single instance and return the result."""
        start_time = datetime.now()
        
        try:
            # Set environment variables for this deployment
            env = os.environ.copy()
            env["MODAL_APP_NAME"] = f"{self.base_app_name}-{instance_number}"
            
            # Deploy the app and capture the output
            result = subprocess.run(
                ["modal", "deploy", str(self.base_app_path)],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            
            # Parse the URL from the deployment output
            url_pattern = r"https://.*?\.modal\.run"
            match = re.search(url_pattern, result.stdout)
            
            if not match:
                raise RuntimeError("Could not find deployment URL in output")
            
            deploy_time = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                instance_number=instance_number,
                url=match.group(0),
                success=True,
                error=None,
                deploy_time=deploy_time,
                timestamp=datetime.now()
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed for instance {instance_number}:\n{e.stderr}")
            deploy_time = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                instance_number=instance_number,
                url=None,
                success=False,
                error=str(e.stderr),
                deploy_time=deploy_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error deploying instance {instance_number}:\n{str(e)}")
            deploy_time = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                instance_number=instance_number,
                url=None,
                success=False,
                error=str(e),
                deploy_time=deploy_time,
                timestamp=datetime.now()
            )

    def deploy_multiple(self, num_instances: int, max_parallel: int = 3) -> List[DeploymentResult]:
        """Deploy multiple instances in parallel."""
        logger.info(f"Starting deployment of {num_instances} instances (max parallel: {max_parallel})...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_instance = {
                executor.submit(self._deploy_single, i + 1): i + 1
                for i in range(num_instances)
            }
            
            for future in as_completed(future_to_instance):
                instance_num = future_to_instance[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.deployments[instance_num] = result
                    
                    if result.success:
                        logger.info(f"✓ Instance {instance_num} deployed successfully: {result.url}")
                    else:
                        logger.error(f"✗ Instance {instance_num} deployment failed:\n{result.error}")
                except Exception as e:
                    logger.error(f"✗ Instance {instance_num} deployment failed with exception:\n{str(e)}")
        
        return sorted(results, key=lambda x: x.instance_number)

def print_deployment_summary(results: List[DeploymentResult]):
    """Print a summary of the deployment results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print("\nDeployment Summary:")
    print("==================")
    print(f"Total instances: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Average deploy time: {sum(r.deploy_time for r in results)/len(results):.2f}s")
    
    if successful:
        print("\nSuccessful Deployments:")
        print("=====================")
        for result in successful:
            print(f"{result.instance_number}. {result.url} ({result.deploy_time:.2f}s)")
    
    if failed:
        print("\nFailed Deployments:")
        print("=================")
        for result in failed:
            print(f"{result.instance_number}. Error:\n{result.error}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python deploy_multiple.py <number_of_instances>")
        sys.exit(1)
        
    try:
        num_instances = int(sys.argv[1])
        if num_instances < 1:
            raise ValueError("Number of instances must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    deployer = ModalDeployer()
    results = deployer.deploy_multiple(num_instances)
    print_deployment_summary(results)

if __name__ == "__main__":
    main() 