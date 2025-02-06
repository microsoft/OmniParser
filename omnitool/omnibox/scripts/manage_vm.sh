#!/bin/bash

create_vm() {
    if ! docker images windows-local -q | grep -q .; then
        echo "Image not found locally. Building..."
        docker build -t windows-local ..
    else
        echo "Image found locally. Skipping build."
    fi

    docker compose -f ../compose.yml up -d

    # Wait for the VM to start up
    while true; do
        response=$(curl --write-out '%{http_code}' --silent --output /dev/null localhost:5000/probe)
        if [ $response -eq 200 ]; then
            break
        fi
        echo "Waiting for a response from the computer control server. When first building the VM storage folder this can take a while..."
        sleep 5
    done

    echo "VM + server is up and running!"
}

start_vm() {
    echo "Starting VM..."
    docker compose -f ../compose.yml start
    while true; do
        response=$(curl --write-out '%{http_code}' --silent --output /dev/null localhost:5000/probe)
        if [ $response -eq 200 ]; then
            break
        fi
        echo "Waiting for a response from the computer control server"
        sleep 5
    done
    echo "VM started"
}

stop_vm() {
    echo "Stopping VM..."
    docker compose -f ../compose.yml stop
    echo "VM stopped"
}

delete_vm() {
    echo "Removing VM and associated containers..."
    docker compose -f ../compose.yml down
    echo "VM removed"
}

# Check if control parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [create|start|stop|delete]"
    exit 1
fi

# Execute the appropriate function based on the control parameter
case "$1" in
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
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 [create|start|stop|delete]"
        exit 1
        ;;
esac