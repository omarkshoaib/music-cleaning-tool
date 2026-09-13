#!/usr/bin/env bash
# Keeps the endpoint serving. Safe to run repeatedly and concurrently.
#
#   keepalive.sh          health-check; restart only after repeated failures
#   keepalive.sh --now    start immediately if not serving (used at boot)
#
# A long job must not be mistaken for a hung server, so an ordinary check has
# to fail FAIL_THRESHOLD times in a row before anything is killed. At the cron
# interval of 2 minutes that is ~6 minutes of genuine unresponsiveness.
set -uo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd "$(dirname "$(readlink -f "$0")")" || exit 1

PORT="${CV_PORT:-8731}"
FAIL_THRESHOLD=3
ENV_NAME="${ENV_NAME:-clearvoice_endpoint}"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

mkdir -p .work logs
LOCKFILE=.work/keepalive.lock
FAILFILE=.work/keepalive.fails

log() { echo "$(date -Is) $*" >> logs/keepalive.log; }

# Heartbeat BEFORE the lock, so every run leaves evidence even if it bails out
# at the lock. Without this a silently-skipping watchdog looks identical to one
# that is never scheduled at all.
date -Is > .work/last_check

exec 9>"$LOCKFILE"
flock -n 9 || exit 0          # another keepalive is already acting

# Health means OUR app is serving, not merely that something answers on the
# port. Another user's Gradio once took this port and returned a perfectly good
# 200, which a liveness-only check accepted as proof of life while our app was
# not running at all. So identity is checked first.
# Identify our server by the socket on OUR port, not just by command name.
# ss only reveals process info for sockets this user owns, so a foreign process
# holding the port yields nothing and is correctly judged "not ours". Matching
# on the command name alone would also match an instance serving a different
# port, and the restart path would then kill the wrong process.
our_server_pid() {
    local pid
    pid=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}$" '$4 ~ p' \
          | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "$pid" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "app.py" || return 1
    echo "$pid"
}

if [ -n "$(our_server_pid)" ] \
   && curl -sf -o /dev/null --max-time 10 "http://127.0.0.1:${PORT}/"; then
    rm -f "$FAILFILE"
    exit 0
fi

if [ "${1:-}" != "--now" ]; then
    fails=$(( $(cat "$FAILFILE" 2>/dev/null || echo 0) + 1 ))
    echo "$fails" > "$FAILFILE"
    if [ "$fails" -lt "$FAIL_THRESHOLD" ]; then
        log "health check failed ($fails/$FAIL_THRESHOLD), waiting"
        exit 0
    fi
fi
rm -f "$FAILFILE"

# Only ever kill our own stale server. A port held by another user's process
# must be reported, not fought over: we cannot signal it, and silently retrying
# forever would look like the watchdog working when nothing is being served.
if ss -ltn 2>/dev/null | grep -q ":${PORT}" && [ -z "$(our_server_pid)" ]; then
    log "PORT CONFLICT: ${PORT} is held by another user's process; cannot start. Set CV_PORT to a free port."
    exit 1
fi

PID=$(our_server_pid)
if [ -n "${PID:-}" ]; then
    log "killing unresponsive server $PID"
    kill "$PID" 2>/dev/null
    for _ in 1 2 3 4 5; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
fi

# shellcheck disable=SC1090
source "$CONDA_SH" || { log "FATAL: no conda at $CONDA_SH"; exit 1; }
conda activate "$ENV_NAME" || { log "FATAL: cannot activate $ENV_NAME"; exit 1; }

# 9>&- is essential. Without it the server inherits the lock file descriptor
# and holds the flock for as long as it runs, so every later keepalive bails at
# the lock and the watchdog silently stops watching. setsid detaches the server
# from this session so it outlives the shell or cron job that started it.
setsid nohup python app.py >> "logs/server-$(date +%Y%m%d).log" 2>&1 9>&- &
started=$!
log "started pid $started on port $PORT"

# A bind failure is silent otherwise: the process exits and the next run just
# tries again forever. Confirm it is actually listening.
sleep 8
if ! kill -0 "$started" 2>/dev/null; then
    log "FAILED: server exited immediately; see logs/server-$(date +%Y%m%d).log"
fi

find logs -name 'server-*.log' -mtime +14 -delete 2>/dev/null
exit 0
