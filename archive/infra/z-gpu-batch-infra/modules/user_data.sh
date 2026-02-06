#!/bin/bash
# Create this file for the launch template

cat > user_data.sh <<'EOF'
#!/bin/bash
set -e

# Install Lustre client
amazon-linux-extras install -y lustre

# Create mount points
mkdir -p /fsx
mkdir -p /fsx-datasets

# Mount FSx Lustre for checkpoints (PERSISTENT)
mount -t lustre -o defaults,_netdev,flock ${fsx_dns_name}@tcp:/${fsx_mount_name} /fsx

# Mount FSx Lustre for datasets (SCRATCH)
mount -t lustre -o defaults,_netdev,flock ${fsx_datasets_dns_name}@tcp:/${fsx_datasets_mount_name} /fsx-datasets

# Add to fstab for persistence
echo "${fsx_dns_name}@tcp:/${fsx_mount_name} /fsx lustre defaults,_netdev,flock 0 0" >> /etc/fstab
echo "${fsx_datasets_dns_name}@tcp:/${fsx_datasets_mount_name} /fsx-datasets lustre defaults,_netdev,flock 0 0" >> /etc/fstab

# Verify mounts
df -h /fsx
df -h /fsx-datasets
EOF