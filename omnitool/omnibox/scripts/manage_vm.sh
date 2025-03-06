#!/bin/bash

# Default VM type is windows if not specified
VM_TYPE="windows"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the parent directory (omnibox directory)
OMNIBOX_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

create_vm() {
    local dockerfile="Dockerfile"
    local compose_file="compose.yml"
    local image_name="windows-local"
    
    if [ "$VM_TYPE" == "linux" ]; then
        dockerfile="Dockerfile.linux"
        compose_file="compose.linux.yml"
        image_name="linux-local"
    fi
    
    if ! docker images $image_name -q | grep -q .; then
        echo "Image not found locally. Building..."
        docker build --no-cache --pull -t $image_name -f "$OMNIBOX_DIR/$dockerfile" "$OMNIBOX_DIR"
    else
        echo "Image found locally. Skipping build."
    fi

    docker compose -f "$OMNIBOX_DIR/$compose_file" up -d

    # Wait for the VM to start up
    while true; do
        response=$(curl --write-out '%{http_code}' --silent --output /dev/null localhost:5000/probe)
        if [ $response -eq 200 ]; then
            break
        fi
        echo "Waiting for a response from the computer control server. When first building the VM storage folder this can take a while..."
        sleep 5
    done

    echo "$VM_TYPE VM + server is up and running!"
}

start_vm() {
    local compose_file="compose.yml"
    
    if [ "$VM_TYPE" == "linux" ]; then
        compose_file="compose.linux.yml"
    fi
    
    echo "Starting $VM_TYPE VM..."
    
    # Show container status before starting
    echo "Container status before starting:"
    docker compose -f "$OMNIBOX_DIR/$compose_file" ps
    
    # Start the containers
    docker compose -f "$OMNIBOX_DIR/$compose_file" start
    
    # Show container status after starting
    echo "Container status after starting:"
    docker compose -f "$OMNIBOX_DIR/$compose_file" ps
    
    # Check if containers are running
    if docker compose -f "$OMNIBOX_DIR/$compose_file" ps --status running | grep -q .; then
        echo "Containers are running."
    else
        echo "Warning: No containers appear to be running. Check logs for details."
        echo "Container logs:"
        docker compose -f "$OMNIBOX_DIR/$compose_file" logs --tail=20
    fi
    
    # Uncomment this section if you want to wait for the server to respond
    while true; do
        response=$(curl --write-out '%{http_code}' --silent --output /dev/null localhost:5000/probe)
        if [ $response -eq 200 ]; then
            break
        fi
        echo "Waiting for a response from the computer control server"
        sleep 5
    done
    
    echo "$VM_TYPE VM started"
}

stop_vm() {
    local compose_file="compose.yml"
    
    if [ "$VM_TYPE" == "linux" ]; then
        compose_file="compose.linux.yml"
    fi
    
    echo "Stopping $VM_TYPE VM..."
    docker compose -f "$OMNIBOX_DIR/$compose_file" stop
    echo "$VM_TYPE VM stopped"
}

delete_vm() {
    local compose_file="compose.yml"
    local image_name="windows-local"
    
    if [ "$VM_TYPE" == "linux" ]; then
        compose_file="compose.linux.yml"
        image_name="linux-local"
    fi
    
    echo "Removing $VM_TYPE VM and associated containers..."
    docker compose -f "$OMNIBOX_DIR/$compose_file" down
    docker rmi $image_name
    echo "$VM_TYPE VM removed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        create|start|stop|delete)
            COMMAND="$1"
            shift
            ;;
        --linux)
            VM_TYPE="linux"
            shift
            ;;
        --windows)
            VM_TYPE="windows"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [create|start|stop|delete] [--linux|--windows]"
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 [create|start|stop|delete] [--linux|--windows]"
    echo "  --linux    Use Linux VM configuration (default is Windows)"
    echo "  --windows  Use Windows VM configuration"
    exit 1
fi

# Execute the appropriate function based on the command
case "$COMMAND" in
    "create")
        create_vm
        ;;
    "start")
        start_vm
        ;;
    "stop")
        stop_vm
        ;;
    "delete")
        delete_vm
        ;;
esac