#!/bin/bash

# 1. Restore the name for GID 984
if getent group 984 > /dev/null; then
    echo "Group ID 984 already has a name."
else
    echo "Fixing: Mapping GID 984 to 'docker'..."
    sudo groupadd -g 984 docker 2>/dev/null || sudo groupmod -n docker $(getent group 984 | cut -d: -f1)
fi

# 2. Add your user to the group
echo "Adding $USER to the docker group..."
sudo usermod -aG docker $USER

# 3. Fix potential permission issues on the group file
echo "Ensuring /etc/group is readable..."
sudo chmod 644 /etc/group

echo "------------------------------------------------"
echo "Done! Please run: 'newgrp docker' to apply now."
echo "Or log out and log back in."
