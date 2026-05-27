#!/usr/bin/env bash
set -euo pipefail

COMPOSE_DIR="${COMPOSE_DIR:-/data/mempalace-compose}"
ENV_FILE="${ENV_FILE:-$COMPOSE_DIR/.env}"
RESTORE_TMP_DIR="${RESTORE_TMP_DIR:-/tmp/mempalace-restore-drill}"
RESTORE_CONTAINER="${RESTORE_CONTAINER:-mempalace-restore-drill}"

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

mkdir -p "$RESTORE_TMP_DIR"
latest="$(rclone lsf "$B2_REMOTE:$B2_BUCKET/" --files-only | grep '^mempalace-.*\.sql\.gz$' | sort | tail -1)"
if [[ -z "$latest" ]]; then
  echo "no backup found in $B2_REMOTE:$B2_BUCKET" >&2
  exit 1
fi

backup_path="$RESTORE_TMP_DIR/$latest"
rclone copyto "$B2_REMOTE:$B2_BUCKET/$latest" "$backup_path"

docker rm -f "$RESTORE_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$RESTORE_CONTAINER" \
  -e POSTGRES_USER=mempalace \
  -e POSTGRES_DB=mempalace \
  -e POSTGRES_PASSWORD=restore-test \
  pgvector/pgvector:pg17 >/dev/null

for _ in {1..30}; do
  if docker exec "$RESTORE_CONTAINER" pg_isready -U mempalace -d mempalace >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
zcat "$backup_path" | docker exec -i "$RESTORE_CONTAINER" psql -U mempalace -d mempalace >/dev/null
row_count="$(docker exec "$RESTORE_CONTAINER" psql -U mempalace -d mempalace -tAc 'SELECT count(*) FROM mempalace_drawers;' 2>/dev/null || echo unknown)"
ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

docker rm -f "$RESTORE_CONTAINER" >/dev/null

printf 'backup=%s\nstarted_at=%s\nended_at=%s\nmempalace_drawers=%s\n' \
  "$latest" "$started_at" "$ended_at" "$row_count"
