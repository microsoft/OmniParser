function Create-VM {
    if (-not (docker images windows-local -q)) {
        Write-Host "Image not found locally. Building..."
        docker build -t windows-local ..
    } else {
        Write-Host "Image found locally. Skipping build."
    }

    docker compose -f ../compose.yml up -d

    while ($true) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/probe" -Method GET -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                break
            }
        } catch {
            Write-Host "Waiting for a response from the computer control server. When first building the VM storage folder this can take a while..."
            Start-Sleep -Seconds 5
        }
    }

    Write-Host "VM + server is up and running!"
}

function Start-LocalVM {
    Write-Host "Starting VM..."
    docker compose -f ../compose.yml start
    while ($true) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/probe" -Method GET -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                break
            }
        } catch {
            Write-Host "Waiting for a response from the computer control server"
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "VM started"
}

function Stop-LocalVM {
    Write-Host "Stopping VM..."
    docker compose -f ../compose.yml stop
    Write-Host "VM stopped"
}

function Remove-VM {
    Write-Host "Removing VM and associated containers..."
    docker compose -f ../compose.yml down
    Write-Host "VM removed"
}

if (-not $args[0]) {
    Write-Host "Usage: $($MyInvocation.MyCommand.Name) [create|start|stop|delete]"
    exit 1
}

switch ($args[0]) {
    "create" { Create-VM }
    "start" { Start-LocalVM }
    "stop" { Stop-LocalVM }
    "delete" { Remove-VM }
    default {
        Write-Host "Invalid option: $($args[0])"
        Write-Host "Usage: $($MyInvocation.MyCommand.Name) [create|start|stop|delete]"
        exit 1
    }
}