#!/usr/bin/env bash
# serve/stop_model.sh <model-key> [--yes] [--timeout SECONDS]
#
# Stops the vLLM server started by serve/serve_model.sh for <model-key>, using
# the pidfile runs/serve/<model-key>.pid. Needed on a single-GPU box, where the
# only way to serve a different model is to stop the current one first.
#
# Check before running this: another job may be generating against the server.
# Without --yes the script only prints what it would kill and exits non-zero.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac
KEY="$1"; shift

YES=0; TIMEOUT=120
while [ $# -gt 0 ]; do
  case "$1" in
    --yes)     YES=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

PIDFILE="$REPO_ROOT/runs/serve/$KEY.pid"
[ -f "$PIDFILE" ] || { echo "no pidfile $PIDFILE - nothing started by serve_model.sh for '$KEY'" >&2; exit 1; }
PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "pid $PID (from $PIDFILE) is not running; removing stale pidfile"
  rm -f "$PIDFILE"; exit 0
fi

echo "would stop: pid $PID"
ps -o pid,etime,rss,args -p "$PID" | sed -n '1,2p'
if [ "$YES" -ne 1 ]; then
  echo "refusing without --yes (never stop a vLLM server someone else may be using)." >&2
  exit 2
fi

kill -INT "$PID" 2>/dev/null || true
for _ in $(seq 1 "$TIMEOUT"); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
if kill -0 "$PID" 2>/dev/null; then
  echo "still alive after ${TIMEOUT}s; sending SIGKILL"
  kill -9 "$PID" 2>/dev/null || true
  sleep 2
fi
rm -f "$PIDFILE"
echo "stopped $KEY (pid $PID); log kept at $REPO_ROOT/runs/serve/$KEY.log"
