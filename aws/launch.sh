#!/usr/bin/env bash
#
# launch.sh — Spin up a GPU spot instance on AWS for CUDA kernel development.
#
# Prerequisites:
#   1. AWS CLI v2 installed:  https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html
#   2. AWS credentials configured:  aws configure
#      (Enter your Access Key ID, Secret Key, region us-east-1, output json)
#   3. If using GitHub Student Pack credits: redeem at https://aws.amazon.com/education/awseducate/
#
# Usage:
#   chmod +x aws/launch.sh
#   ./aws/launch.sh              # launches g5.xlarge spot instance
#   ./aws/launch.sh g4dn.xlarge  # cheaper T4 option (~$0.16/hr spot)
#
# After launch, it prints an SSH command. Run it, then run aws/setup_remote.sh on the instance.
# When done: ./aws/teardown.sh
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
INSTANCE_TYPE="${1:-g5.xlarge}"      # g5.xlarge = A10G ($~1/hr spot), g4dn.xlarge = T4 (~$0.16/hr spot)
REGION="us-east-1"
KEY_NAME="fused-attention-key"
SG_NAME="fused-attention-sg"
TAG="fused-attention"

echo "=== Fused Attention — AWS GPU Instance Launcher ==="
echo "Instance type: $INSTANCE_TYPE"
echo "Region:        $REGION"
echo ""

# ── Step 1: Look up the Deep Learning AMI ────────────────────────────────────
echo "[1/5] Looking up Deep Learning AMI..."
# Use the Deep Learning Base GPU AMI (Ubuntu 22.04) — has CUDA 12.x, nvcc, ncu pre-installed
AMI_ID=$(aws ssm get-parameter \
    --region "$REGION" \
    --name "/aws/service/deeplearning/ami/x86_64/base-gpu-ubuntu-22-04" \
    --query "Parameter.Value" \
    --output text 2>/dev/null || true)

if [ -z "$AMI_ID" ]; then
    echo "  SSM lookup failed, falling back to manual search..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=*Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: Could not find Deep Learning AMI. Check your AWS credentials and region."
    exit 1
fi
echo "  AMI: $AMI_ID"

# ── Step 2: Create a key pair (if it doesn't exist) ─────────────────────────
echo "[2/5] Setting up SSH key pair..."
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
if aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" &>/dev/null; then
    echo "  Key pair '$KEY_NAME' already exists."
    if [ ! -f "$KEY_FILE" ]; then
        echo "  WARNING: Key file $KEY_FILE not found locally."
        echo "  Either find it or delete the key pair and re-run:"
        echo "    aws ec2 delete-key-pair --region $REGION --key-name $KEY_NAME"
        exit 1
    fi
else
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query "KeyMaterial" \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "  Created $KEY_FILE"
fi

# ── Step 3: Create a security group (if it doesn't exist) ───────────────────
echo "[3/5] Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "SSH access for fused-attention GPU dev" \
        --query "GroupId" \
        --output text)

    # Allow SSH from your current IP only
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr "${MY_IP}/32"
    echo "  Created SG $SG_ID, SSH allowed from $MY_IP"
else
    echo "  Security group '$SG_NAME' already exists ($SG_ID)."
fi

# ── Step 4: Request a spot instance ──────────────────────────────────────────
echo "[4/5] Launching spot instance..."

# Use a launch specification for spot request
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  Instance ID: $INSTANCE_ID"
echo "  Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# ── Step 5: Get the public IP ────────────────────────────────────────────────
echo "[5/5] Getting connection info..."
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

# Save instance info for teardown
STATE_FILE="$(dirname "$0")/.instance-state"
cat > "$STATE_FILE" <<EOF
INSTANCE_ID=$INSTANCE_ID
REGION=$REGION
KEY_NAME=$KEY_NAME
SG_NAME=$SG_NAME
SG_ID=$SG_ID
KEY_FILE=$KEY_FILE
PUBLIC_IP=$PUBLIC_IP
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Instance is running!"
echo ""
echo "  SSH in (may take 1-2 min for sshd to start):"
echo ""
echo "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo ""
echo "  Then on the instance, run:"
echo ""
echo "    # Copy setup script and project files"
echo "    # (from your local machine, in another terminal):"
echo "    scp -i $KEY_FILE -r $(dirname "$0")/../* ubuntu@${PUBLIC_IP}:~/fused-attention/"
echo "    scp -i $KEY_FILE $(dirname "$0")/setup_remote.sh ubuntu@${PUBLIC_IP}:~/"
echo ""
echo "    # Then on the instance:"
echo "    chmod +x ~/setup_remote.sh && ~/setup_remote.sh"
echo ""
echo "  When done, TERMINATE to stop charges:"
echo ""
echo "    ./aws/teardown.sh"
echo ""
echo "  Estimated cost: ~\$${INSTANCE_TYPE##g}"
if [ "$INSTANCE_TYPE" = "g5.xlarge" ]; then
    echo "  (g5.xlarge spot ≈ \$0.50-1.00/hr — A10G, good for benchmarks)"
elif [ "$INSTANCE_TYPE" = "g4dn.xlarge" ]; then
    echo "  (g4dn.xlarge spot ≈ \$0.16/hr — T4, cheaper but weaker)"
fi
echo "════════════════════════════════════════════════════════════════"
