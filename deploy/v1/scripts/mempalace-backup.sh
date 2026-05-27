#!/usr/bin/env bash
set -euo pipefail

COMPOSE_DIR="${COMPOSE_DIR:-/data/mempalace-compose}"
ENV_FILE="${ENV_FILE:-$COMPOSE_DIR/.env}"
BACKUP_TMP_DIR="${BACKUP_TMP_DIR:-/tmp/mempalace-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${B2_REMOTE:?B2_REMOTE is required}"
: "${B2_BUCKET:?B2_BUCKET is required}"

mkdir -p "$BACKUP_TMP_DIR"
stamp="$(date -u +%Y-%m-%dT%H%M%SZ)"
local_path="$BACKUP_TMP_DIR/mempalace-$stamp.sql.gz"
remote_path="$B2_REMOTE:$B2_BUCKET/mempalace-$stamp.sql.gz"

cd "$COMPOSE_DIR"
docker compose exec -T mempalace-postgres \
  pg_dump -U mempalace -d mempalace \
  | gzip -9 > "$local_path"

rclone copyto "$local_path" "$remote_path"
rclone lsf "$B2_REMOTE:$B2_BUCKET/" --files-only --max-age "${RETENTION_DAYS}d" >/dev/null
find "$BACKUP_TMP_DIR" -type f -name 'mempalace-*.sql.gz' -mtime +2 -delete

echo "$remote_path"
