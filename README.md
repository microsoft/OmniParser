# OmniParserV2 Modal Cluster

[![License: AGPL](https://img.shields.io/badge/License-AGPL-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OmniParserV2 Modal Cluster is a scalable, production-ready fork of [Microsoft's OmniParser](https://github.com/microsoft/OmniParser) that is optimized for cloud deployments using [Modal Labs](https://modal.com/). This project is designed for horizontal scaling, enabling multiple parallel instances to handle concurrent requests efficiently and process large volumes of data quickly.

## Key Features

- **Horizontal Scaling:** Deploy multiple instances to handle concurrent requests with dedicated GPU acceleration.
- **Cloud-Native Optimization:** Leverage Modal Labs' infrastructure to build efficient, scalable, and cost-effective cloud applications.
- **Enhanced Image Processing:** Supports base64-encoded images and direct file uploads for versatile image input handling.
- **Production-Ready:** Incorporates robust error handling and optimal resource management to ensure reliability.

## Architecture

The project consists of several core components:

- **[modal_app.py](modal_app.py):** Configures the deployment environment and GPU acceleration using Modal Labs.
- **[deploy_multiple.py](deploy_multiple.py):** Orchestrates the deployment of multiple OmniParser Modalinstances in parallel.
- **[gradio_demo.py](gradio_demo.py):** Provides an interactive demo interface with advanced image processing capabilities.

Each instance operates independently, ensuring isolation and dedicated GPU resources, which results in improved stability and performance in a cloud-serverless environment.

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

## Configuration

Before deployment, configure the following environment variables as per your requirements:

| Variable                      | Description                                          | Default         |
| ----------------------------- | ---------------------------------------------------- | --------------- |
| `MODAL_APP_NAME`              | Base name for the Modal application                  | "omniparser-v2" |
| `MODAL_GPU_CONFIG`            | GPU type for instances                               | "L4"          |
| `MODAL_CONTAINER_TIMEOUT`     | Container idle timeout (in seconds)                  | 120             |
| `MODAL_MAX_CONCURRENT_INPUTS` | Maximum number of queued inputs                      | 50              |
| `MODAL_CONCURRENCY_LIMIT`     | Maximum concurrent requests per instance             | 1               |
| `GRADIO_PORT`                 | Port number for the Gradio interface                 | 7860            |

Example configuration:
```bash
export MODAL_APP_NAME="omniparser"
export MODAL_GPU_CONFIG="T4"
export MODAL_CONTAINER_TIMEOUT=300
export MODAL_MAX_CONCURRENT_INPUTS=100
export MODAL_CONCURRENCY_LIMIT=1
export GRADIO_PORT=8000

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
