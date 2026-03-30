read_config_value() {
  local config_path="$1"
  local key="$2"
  "${PYTHON_BIN}" - "$config_path" "$key" <<'PYCFG'
import json
import sys
from pathlib import Path
config_path = Path(sys.argv[1])
key = sys.argv[2]
try:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
value = payload.get(key, "")
if value is None:
    value = ""
print(value)
PYCFG
}

local_videobench_server_pid=""

stop_local_videobench_server() {
  if [ -n "${local_videobench_server_pid:-}" ] && kill -0 "${local_videobench_server_pid}" >/dev/null 2>&1; then
    kill "${local_videobench_server_pid}" >/dev/null 2>&1 || true
    wait "${local_videobench_server_pid}" 2>/dev/null || true
  fi
}

wait_for_local_videobench_server() {
  local base_url="$1"
  local timeout_s="$2"
  local poll_s="$3"
  local elapsed=0
  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    if "${PYTHON_BIN}" - "$base_url" <<'PYWAIT'
import json
import sys
import urllib.request
base_url = sys.argv[1].rstrip('/')
url = base_url + '/models'
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode('utf-8', errors='replace'))
    raise SystemExit(0 if isinstance(payload.get('data', []), list) else 1)
except Exception:
    raise SystemExit(1)
PYWAIT
    then
      return 0
    fi
    sleep "${poll_s}"
    elapsed=$((elapsed + poll_s))
  done
  return 1
}

maybe_start_local_videobench_server() {
  local config_path="$1"
  [ -f "${config_path}" ] || return 0

  local base_url host port model_name api_key
  local local_model_id local_model_dir local_snapshot_root local_download_model local_extra_pips
  base_url="$(read_config_value "${config_path}" GPT4o_BASE_URL)"
  [ -n "${base_url}" ] || base_url="$(read_config_value "${config_path}" OPENAI_BASE_URL)"
  [ -n "${base_url}" ] || return 0

  host="$("${PYTHON_BIN}" - "$base_url" <<'PYHOST'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
print(u.hostname or '')
PYHOST
)"
  port="$("${PYTHON_BIN}" - "$base_url" <<'PYPORT'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1])
print(u.port or (443 if u.scheme == 'https' else 80))
PYPORT
)"

  if [ "${START_LOCAL_VIDEOBENCH_SERVER:-auto}" = "0" ]; then
    return 0
  fi
  if [ "${START_LOCAL_VIDEOBENCH_SERVER:-auto}" = "auto" ] && [ "${host}" != "127.0.0.1" ] && [ "${host}" != "localhost" ]; then
    return 0
  fi

  if wait_for_local_videobench_server "${base_url}" 2 1; then
    echo "Local Video-Bench server already reachable at ${base_url}"
    return 0
  fi

  if [ ! -x "${VIDEOBENCH_LOCAL_SERVER_SCRIPT}" ]; then
    echo "Local Video-Bench config points to ${base_url}, but launcher is missing or not executable: ${VIDEOBENCH_LOCAL_SERVER_SCRIPT}"
    exit 1
  fi

  model_name="$(read_config_value "${config_path}" GPT4o_MODEL)"
  [ -n "${model_name}" ] || model_name="$(read_config_value "${config_path}" OPENAI_MODEL)"
  api_key="$(read_config_value "${config_path}" GPT4o_API_KEY)"
  [ -n "${api_key}" ] || api_key="$(read_config_value "${config_path}" OPENAI_API_KEY)"
  [ -n "${api_key}" ] || api_key="${LOCAL_SERVER_API_KEY:-local-videobench}"
  local_model_id="$(read_config_value "${config_path}" LOCAL_SERVER_MODEL_ID)"
  local_model_dir="$(read_config_value "${config_path}" LOCAL_SERVER_MODEL_DIR)"
  local_snapshot_root="$(read_config_value "${config_path}" LOCAL_SERVER_HF_CACHE_SNAPSHOT_ROOT)"
  local_download_model="$(read_config_value "${config_path}" LOCAL_SERVER_DOWNLOAD_MODEL)"
  local_extra_pips="$(read_config_value "${config_path}" LOCAL_SERVER_EXTRA_PIP_PACKAGES)"

  mkdir -p "${WINDOW_REPORT_ROOT}"
  local server_log="${WINDOW_REPORT_ROOT}/local_videobench_server_${RUN_NAME_BASE}.log"
  echo "Starting local Video-Bench server at ${base_url}"
  HOST="${host}" \
  PORT="${port}" \
  API_KEY="${api_key}" \
  SERVED_MODEL_NAME="${model_name:-Qwen2.5-VL-7B-Instruct}" \
  MODEL_ID="${local_model_id}" \
  MODEL_DIR="${local_model_dir}" \
  HF_CACHE_SNAPSHOT_ROOT="${local_snapshot_root}" \
  DOWNLOAD_MODEL="${local_download_model:-1}" \
  SERVER_EXTRA_PIP_PACKAGES="${local_extra_pips}" \
  bash "${VIDEOBENCH_LOCAL_SERVER_SCRIPT}" >"${server_log}" 2>&1 &
  local_videobench_server_pid=$!
  trap stop_local_videobench_server EXIT

  if ! wait_for_local_videobench_server "${base_url}" "${LOCAL_SERVER_WAIT_SECONDS:-240}" "${LOCAL_SERVER_POLL_INTERVAL:-2}"; then
    echo "Local Video-Bench server did not become ready at ${base_url}"
    echo "Server log: ${server_log}"
    exit 1
  fi

  echo "Local Video-Bench server is ready at ${base_url}"
}
