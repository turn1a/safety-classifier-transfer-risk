#!/bin/bash
# Minimal first-boot bootstrap (rendered by Terraform templatefile — only the ${...} below are
# substituted; it deliberately uses no bash variables, so nothing needs $$-escaping). It writes
# the run config, clones the repo, and hands off to the committed infra/cloud_run.sh (a normal
# bash script) which installs the project and runs the sweep — reading inputs and writing
# partitions through the Kedro catalog (S3), with no aws s3 sync — as ec2-user.
set -euo pipefail
exec > >(tee -a /var/log/tr-bootstrap.log) 2>&1

# Safety net: auto-shutdown after the max lifetime so a hung or idle box cannot bill forever.
shutdown -h +${max_lifetime} "transfer-risk: max lifetime reached" || true

# Stream the log to S3 every 30s for observability without opening a session.
( while true; do aws s3 cp /var/log/tr-bootstrap.log s3://${bucket}/logs/latest/run.log --region ${region} >/dev/null 2>&1; sleep 30; done ) &

dnf -y install git gcc gcc-c++ make tmux

cat > /opt/config.env <<EOF
export TR_BUCKET='${bucket}'
export TR_REGION='${region}'
export TR_REPO_REF='${repo_ref}'
EOF

runuser -l ec2-user -c 'git clone ${repo_url} ~/repo && bash ~/repo/infra/cloud_run.sh' || true

aws s3 cp /var/log/tr-bootstrap.log s3://${bucket}/logs/latest/run.log --region ${region} || true
shutdown -h now
