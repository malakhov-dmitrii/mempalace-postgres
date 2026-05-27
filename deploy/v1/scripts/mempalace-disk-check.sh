#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/data/mempalace-compose/.env}"
MOUNT_PATH="${MOUNT_PATH:-/data}"
THRESHOLD="${THRESHOLD:-85}"

if [[ -r "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${UPTIME_KUMA_DISK_PUSH_URL:?UPTIME_KUMA_DISK_PUSH_URL is required}"

used="$(df -P "$MOUNT_PATH" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
if [[ "$used" -ge "$THRESHOLD" ]]; then
  curl -fsS "${UPTIME_KUMA_DISK_PUSH_URL}?status=down&msg=disk_${used}_percent" >/dev/null
else
  curl -fsS "${UPTIME_KUMA_DISK_PUSH_URL}?status=up&msg=disk_${used}_percent" >/dev/null
fi
