# OmniParserV2 Modal Cluster

[![License: AGPL](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OmniParserV2 Modal Cluster is a scalable, production-ready fork of [Microsoft's OmniParser](https://github.com/microsoft/OmniParser) that is optimized for cloud deployments using [Modal Labs](https://modal.com/). This project is designed for horizontal scaling, enabling multiple parallel instances to handle concurrent requests efficiently and process large volumes of data quickly.

## Key Features

- **Horizontal Scaling:** Deploy multiple instances to handle concurrent requests with dedicated GPU acceleration.
- **Cloud-Native Optimization:** Leverage Modal Labs' infrastructure to build efficient, scalable, and cost-effective cloud applications.
- **Enhanced Image Processing:** Supports base64-encoded images and direct file uploads for versatile image input handling.
- **Thread-Safe OCR:** Mitigates segfaults from concurrent requests by incorporating a thread lock in PaddleOCR calls.
- **Production-Ready:** Incorporates robust error handling and optimal resource management to ensure reliability.

## Changes from the original OmniParser

- **[modal_app.py](modal_app.py):** Configures the deployment environment and GPU acceleration using Modal Labs.
- **[deploy_multiple.py](deploy_multiple.py):** Orchestrates the deployment of multiple OmniParser Modal instances in parallel.
- **[gradio_demo.py](gradio_demo.py):** Provides an interactive demo interface with advanced image processing capabilities, including thread-safe OCR with thread locking.
- **[util/utils.py](util/utils.py):** Implements utility functions with thread locking to ensure safe concurrent execution.

Each instance operates independently, ensuring isolation and dedicated GPU resources, which results in improved stability and performance in a cloud-serverless environment.

## Gradio and Modal Scaling Considerations

While Modal offers [concurrent input handling](https://modal.com/docs/guide/concurrent-inputs) within a single container, this project intentionally uses multiple separate instances instead. This design choice is necessary because Gradio applications make several sequential requests (upload, queue status check, and process) that must be handled by the same instance to maintain session state. Modal's automatic container scaling could route these related requests to different containers, breaking the Gradio workflow.

To ensure reliable operation:
- Each Gradio interface runs as a separate Modal app instance
- Requests from a single Gradio session are guaranteed to hit the same backend instance
- Scaling is achieved by creating multiple complete instances rather than scaling containers within a single instance
- This approach trades some resource efficiency for guaranteed session consistency

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install modal
   ```

2. **Set Up Modal Authentication:**
   ```bash
   modal token new
   ```

3. **Deploy Instances:**
   Replace `<number_of_instances>` with the desired number of instances (up to 8 for Modal's free tier):
   ```bash
   python deploy_multiple.py <number_of_instances>
   ```

4. **Stop All Deployed Instances:**
   After you are done with your deployments and no longer need the instances running, you can stop all deployed instances using the following command:
   ```bash
   modal app list --json | jq -r '.[] | select(.State=="deployed") | .["App ID"]' | xargs -n1 modal app stop
   ```

## Configuration

Before deployment, configure the following environment variables as per your requirements:

| Variable                      | Description                                          | Default         |
| ----------------------------- | ---------------------------------------------------- | --------------- |
| `CONCURRENCY_LIMIT`           | Maximum concurrent requests per instance             | 50              |
| `GRADIO_PORT`                 | Port number for the Gradio interface                 | 7860            |
| `MODAL_APP_NAME`              | Base name for the Modal application                  | "omniparser"    |
| `MODAL_CONCURRENT_CONTAINERS` | Maximum number of concurrent containers per instance | 1               |
| `MODAL_CONTAINER_TIMEOUT`     | Container idle timeout (in seconds)                  | 120             |
| `MODAL_GPU_CONFIG`            | GPU type for instances                               | "T4"            |

Example configuration:
```bash
export CONCURRENCY_LIMIT=50
export GRADIO_PORT=7860
export MODAL_APP_NAME="omniparser"
export MODAL_CONCURRENT_CONTAINERS=1
export MODAL_CONTAINER_TIMEOUT=120
export MODAL_GPU_CONFIG="T4"

python deploy_multiple.py 8
```

## Performance Considerations

- Modal's free tier supports a maximum of 8 concurrent web endpoints. For increased capacity, consider upgrading your Modal plan.
- Optimize performance by scaling horizontally—deploy additional instances rather than increasing concurrent requests to a single instance.
- GPU memory usage will scale with the number of instances and the size of the processed images.
- For optimal results, use a pool of clients that intelligently balances load across instances to minimize pending requests on each instance, rather than a simple round robin approach.


## Known Limitations

- Retain `MODAL_CONCURRENCY_LIMIT=1` as the underlying model is not thread-safe.
- Each instance handles only one request at a time; concurrent request handling is achieved through multiple instance deployments.
- Cold starts may occur if a container exceeds the specified idle timeout.

## License
This project is dual-licensed under the same licenses as the original:
- **Icon Detection Model:** [AGPL License](https://www.gnu.org/licenses/agpl-3.0)
- **Icon Caption Models (BLIP2 & Florence):** [MIT License](https://opensource.org/licenses/MIT)

## Acknowledgments

- The original OmniParser team at Microsoft for the base project.
- Modal Labs for providing robust cloud deployment infrastructure.
