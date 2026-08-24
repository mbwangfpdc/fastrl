# Watchdog runner for config probes.
#
# A bad config does NOT fail fast here: the actor dies, the Ray driver keeps
# waiting, and a plain `timeout` then burns its full budget (we lost 4 x 40min
# that way). This polls the log and kills the moment the outcome is decided:
#   fatal marker  -> a Python/Ray error that will never recover
#   completion    -> the step count we asked for
#   stall         -> no log growth for STALL_SEC (the actor-death-then-hang case,
#                    which produces no further output at all)
#   deadline      -> absolute backstop
#
# Anything killed is reaped, including the engine subprocesses, which do not die
# with the driver and will hold the cards against the next probe.

FATAL_RE='ActorUnavailableError|ActorDiedError|RayTaskError|SYSTEM_ERROR|CUDA out of memory|Not enough memory|AssertionError:|Traceback \(most recent call last\)'

reap_engines () {
  pkill -9 -u "$USER" -f "sglang::"        2>/dev/null
  pkill -9 -u "$USER" -f "ray::WorkerDict" 2>/dev/null
  pkill -9 -u "$USER" -f "main_fastrl"     2>/dev/null
  sleep "${REAP_SEC:-40}"
}

# probe_run <name> <logdir> <steps> <max_sec> <stall_sec> -- <env assignments...>
probe_run () {
  local name=$1 logdir=$2 steps=$3 max_sec=$4 stall_sec=$5; shift 6
  local log="$logdir/$name.log"
  if grep -q "global_step:$steps" "$log" 2>/dev/null; then echo "== $name: done, skip"; return 0; fi
  reap_engines
  echo "== $name: $*"
  : > "$log"
  ( cd "$FASTRL_ROOT" && env CUDA_VISIBLE_DEVICES=0,1,2,3 "$@" \
      bash examples/grpo_sql_baseline.sh >> "$log" 2>&1 ) &
  local pid=$! t0=$SECONDS last_sz=0 last_chg=$SECONDS verdict=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    if grep -qE "$FATAL_RE" "$log" 2>/dev/null; then verdict="fatal"; break; fi
    if grep -q "global_step:$steps" "$log" 2>/dev/null; then verdict="ok"; break; fi
    local sz; sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$sz" != "$last_sz" ]; then last_sz=$sz; last_chg=$SECONDS; fi
    if [ $((SECONDS - last_chg)) -gt "$stall_sec" ]; then verdict="stalled"; break; fi
    if [ $((SECONDS - t0))    -gt "$max_sec"   ]; then verdict="deadline"; break; fi
  done
  kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  local elapsed=$((SECONDS - t0))
  local ua ol gn st
  ua=$(grep -oE "timing_s/update_actor:[0-9.]+"       "$log" | tail -1 | cut -d: -f2)
  ol=$(grep -oE "timing_s/old_log_prob:[0-9.]+"       "$log" | tail -1 | cut -d: -f2)
  gn=$(grep -oE "timing_s/generate_sequences:[0-9.]+" "$log" | tail -1 | cut -d: -f2)
  st=$(grep -oE "timing_s/step:[0-9.]+"               "$log" | tail -1 | cut -d: -f2)
  if [ -n "$ua" ]; then
    printf "   OK   update_actor %-8.1f old_log_prob %-8.1f gen %-8.1f step %-8.1f  [%ss]\n" \
           "$ua" "$ol" "$gn" "$st" "$elapsed"
  else
    local why="$verdict"
    grep -q "CUDA out of memory" "$log" 2>/dev/null && why="$why:OOM"
    grep -q "Not enough memory"  "$log" 2>/dev/null && why="$why:sglang-init"
    grep -qE "ActorUnavailableError|ActorDiedError|SYSTEM_ERROR" "$log" 2>/dev/null && why="$why:actor-died"
    grep -q "AssertionError:" "$log" 2>/dev/null && why="$why:assert"
    printf "   FAIL (%s) after %ss\n" "$why" "$elapsed"
  fi
}
