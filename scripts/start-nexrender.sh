#!/bin/bash
# Start Nexrender server + worker for local AE rendering
# Usage: ./scripts/start-nexrender.sh

set -e

SECRET="docket-ae-render-2026"
PORT=3050
AERENDER="/Applications/Adobe After Effects 2025/aerender"

# Check After Effects
if [ ! -f "$AERENDER" ]; then
    # Try other common paths
    AERENDER=$(find /Applications -name "aerender" -type f 2>/dev/null | head -1)
    if [ -z "$AERENDER" ]; then
        echo "ERROR: After Effects not found. Install it from Creative Cloud."
        echo "Expected: /Applications/Adobe After Effects 2025/aerender"
        exit 1
    fi
fi

echo "After Effects: $AERENDER"
echo "Server: http://localhost:$PORT"
echo ""

# Start server in background
echo "Starting nexrender-server on port $PORT..."
nexrender-server --port $PORT --secret $SECRET &
SERVER_PID=$!

# Give server a moment to bind
sleep 2

# Start worker
echo "Starting nexrender-worker..."
nexrender-worker \
    --host "http://localhost:$PORT" \
    --secret $SECRET \
    --binary "$AERENDER" \
    --workpath "/tmp/nexrender" &
WORKER_PID=$!

echo ""
echo "Nexrender running (server PID: $SERVER_PID, worker PID: $WORKER_PID)"
echo "Press Ctrl+C to stop both."

# Trap Ctrl+C to kill both
trap "kill $SERVER_PID $WORKER_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT TERM

# Wait for either to exit
wait
