#!/usr/bin/env bash
#
# teardown.sh — Terminate the GPU instance and clean up AWS resources.
#
# RUN THIS WHEN YOU'RE DONE. Spot instances still charge until terminated.
#
# Usage:
#   ./aws/teardown.sh          # terminate instance, keep key+SG for next time
#   ./aws/teardown.sh --full   # terminate instance AND delete key pair + security group
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="$SCRIPT_DIR/.instance-state"

if [ ! -f "$STATE_FILE" ]; then
    echo "No instance state file found at $STATE_FILE"
    echo "Either the instance was never launched or already torn down."
    echo ""
    echo "To manually find and terminate instances:"
    echo "  aws ec2 describe-instances --filters 'Name=tag:Name,Values=fused-attention' \\"
    echo "    --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table"
    exit 1
fi

source "$STATE_FILE"

echo "=== Fused Attention — Teardown ==="
echo "Instance:  $INSTANCE_ID"
echo "Region:    $REGION"
echo ""

# ── Terminate the instance ───────────────────────────────────────────────────
echo "[1] Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "TerminatingInstances[0].CurrentState.Name" \
    --output text
echo "  Termination initiated. Charges stop within a few minutes."

# ── Full cleanup (optional) ──────────────────────────────────────────────────
if [ "${1:-}" = "--full" ]; then
    echo ""
    echo "[2] Waiting for instance to terminate..."
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID" 2>/dev/null || true

    echo "[3] Deleting security group $SG_NAME ($SG_ID)..."
    aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null && \
        echo "  Deleted." || echo "  Already deleted or in use."

    echo "[4] Deleting key pair $KEY_NAME..."
    aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null && \
        echo "  Deleted from AWS." || echo "  Already deleted."
    if [ -f "$KEY_FILE" ]; then
        rm -f "$KEY_FILE"
        echo "  Deleted local key file $KEY_FILE"
    fi
else
    echo ""
    echo "  Key pair and security group kept for next time."
    echo "  Run with --full to delete those too."
fi

rm -f "$STATE_FILE"

echo ""
echo "Done. No more charges accruing."
