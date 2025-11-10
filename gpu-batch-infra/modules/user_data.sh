#!/bin/bash
# Create this file for the launch template

cat > user_data.sh <<'EOF'
#!/bin/bash
set -e

# Install Lustre client
amazon-linux-extras install -y lustre

# Create mount point
mkdir -p /fsx

# Mount FSx Lustre
mount -t lustre -o defaults,_netdev,flock ${fsx_dns_name}@tcp:/${fsx_mount_name} /fsx

# Add to fstab for persistence
echo "${fsx_dns_name}@tcp:/${fsx_mount_name} /fsx lustre defaults,_netdev,flock 0 0" >> /etc/fstab

# Verify mount
df -h /fsx
EOF