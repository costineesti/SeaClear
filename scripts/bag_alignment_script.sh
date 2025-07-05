#!/bin/bash

# Script to analyze and align rosbag timing

echo "Analyzing bag timing..."

# Get info from both bags
echo "=== Depth Bag Info ==="
rosbag info ~/Downloads/Data_2025-06-23-13-54-24.bag | grep -E "(start|end|duration)"

echo ""
echo "=== GoPro Bag Info ==="
rosbag info ~/Desktop/23Jun_GoPro.bag | grep -E "(start|end|duration)"

echo ""
echo "=== Computing automatic offset ==="

# Extract start timestamps - use the Unix timestamp in parentheses
DEPTH_TIMESTAMP=$(rosbag info ~/Downloads/Data_2025-06-23-13-54-24.bag | grep "start:" | sed 's/.*(\(.*\))/\1/')
GOPRO_TIMESTAMP=$(rosbag info ~/Desktop/23Jun_GoPro.bag | grep "start:" | sed 's/.*(\(.*\))/\1/')

echo "Depth bag starts at: $DEPTH_TIMESTAMP"
echo "GoPro bag starts at: $GOPRO_TIMESTAMP"

# Calculate offset
OFFSET=$(echo "$DEPTH_TIMESTAMP - $GOPRO_TIMESTAMP" | bc -l)
ABS_OFFSET=$(echo "sqrt($OFFSET * $OFFSET)" | bc -l)

echo ""
echo "Time offset: $OFFSET seconds"
echo "Absolute offset: $ABS_OFFSET seconds"

# Determine synchronization strategy
if (( $(echo "$OFFSET > 0" | bc -l) )); then
    echo "Depth bag starts first"
    echo ""
    echo "=== Synchronized Playback Command ==="
    echo "rosbag play --loop --clock ~/Downloads/Data_2025-06-23-13-54-24.bag ~/Desktop/23Jun_GoPro.bag"
else
    START_TIME=$(echo "0 - $OFFSET" | bc -l)
    echo "GoPro bag starts first - need to delay depth bag by $START_TIME seconds"
    echo ""
    echo "=== Synchronized Playback Command ==="
    echo "rosbag play --loop --clock --start=$START_TIME ~/Downloads/Data_2025-06-23-13-54-24.bag ~/Desktop/23Jun_GoPro.bag"
fi

echo ""
echo "=== Topic timing analysis ==="
echo "Checking topics from each bag..."

# Check topics and message counts
echo "Depth bag topics:"
rosbag info ~/Downloads/Data_2025-06-23-13-54-24.bag | grep -A 10 "topics:"

echo ""
echo "GoPro bag topics:"
rosbag info ~/Desktop/23Jun_GoPro.bag | grep -A 10 "topics:"

echo ""
echo "=== Ready to use synchronized command above ==="