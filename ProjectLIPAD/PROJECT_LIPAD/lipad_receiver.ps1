# ====================================================================
# PROJECT LIPAD: GROUND STATION TELEMETRY RECEIVER
# ====================================================================
Clear-Host
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         PROJECT LIPAD - GROUND STATION INTERFACE         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Target Subnet IP : 10.42.0.255 (Hotspot Broadcast)" -ForegroundColor Yellow
Write-Host "Listening Port   : 50007 [UDP]" -ForegroundColor Yellow
Write-Host "Status           : Initializing Socket... Press CTRL+C to Exit." -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# Define Network Configurations
$Port = 50007
$UDPClient = New-Object System.Net.Sockets.UdpClient($Port)
$RemoteEndPoint = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, 0)

Write-Host "--> Port 50007 Clear. Waiting for incoming drone telemetry..." -ForegroundColor Green
Write-Host ""

try {
    while ($true) {
        # Block and wait for incoming UDP broadcast data packets
        $ByteData = $UDPClient.Receive([ref]$RemoteEndPoint)
        $Message = [System.Text.Encoding]::UTF8.GetString($ByteData)
        
        # Format the time the packet hit the laptop ground station
        $Timestamp = Get-Date -Format "HH:mm:ss.fff"
        
        Write-Output "[$Timestamp] FROM: $($RemoteEndPoint.Address) --> $Message"
    }
}
catch [System.Management.Automation.PipelineStoppedException] {
    Write-Host "`nListener stopped cleanly by user." -ForegroundColor Yellow
}
catch {
    # If the socket object exists, display the error nicely
    if ($UDPClient) {
        Write-Host "`nData Stream interrupted: $_" -ForegroundColor Red
    }
}
finally {
    # Only try to close the socket if it was actually created
    if ($null -ne $UDPClient) {
        $UDPClient.Close()
        Write-Host "UDP Socket Closed Safely. System Standby." -ForegroundColor Yellow
    }
}