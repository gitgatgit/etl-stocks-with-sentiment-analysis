#!/bin/bash

# Define the missing IDs and their standard names
# Based on your error output
declare -A GROUPS_TO_FIX
GROUPS_TO_FIX=( [4]="adm" [24]="cdrom" [30]="dip" [984]="docker" )

echo "Starting group name recovery..."

for gid in "${!GROUPS_TO_FIX[@]}"; do
    group_name="${GROUPS_TO_FIX[$gid]}"
    
    # Check if the group name already exists
    if getent group "$group_name" > /dev/null; then
        echo "OK: Group '$group_name' already exists."
    else
        echo "FIXING: Adding group '$group_name' with GID $gid..."
        sudo groupadd -g "$gid" "$group_name"
    fi
done

# Ensure your current user is added to the newly restored docker group
echo "Adding $USER to docker group..."
sudo usermod -aG docker $USER

echo "Done. Please log out and log back in for changes to take effect."
