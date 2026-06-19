"""
╔═══════════════════════════════════════════════════════╗
║     SYSTEM MONITOR — Python Backend (server.py)       ║
║     Flask + WebSocket + psutil                        ║
║     Tutorial by: Chapter 10-12                        ║
╚═══════════════════════════════════════════════════════╝

Install karo pehle:
  pip install flask flask-sock psutil flask-cors dxcam opencv-python

Chalao:
  python server.py

Phir browser mein: http://localhost:5000/api/stats
Ya index.html kholo Live Server se.
"""

# ═══ IMPORTS ═══
from flask import Flask, jsonify
from flask_sock import Sock
from flask_cors import CORS
import psutil
import json
import time
import platform
import logging
from collections import deque
import subprocess
import shlex
import shutil
import os
try:
    import dxcam
except ImportError:  # pragma: no cover - optional runtime dependency
    dxcam = None

# Try NVML (NVIDIA) first for robust GPU metrics
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pynvml = None
    NVML_AVAILABLE = False

# ═══ APP SETUP ═══
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
sock = Sock(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

GPU_UTIL_HISTORY = {}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_rolling_average(samples, window=8):
    """Return the average of the most recent samples, with a safe fallback."""
    if not samples:
        return 0.0
    values = list(samples)[-window:]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def detect_vendor(name):
    """Infer the GPU vendor from a display name."""
    text = (name or '').lower()
    if any(keyword in text for keyword in ('nvidia', 'geforce', 'quadro', 'tesla')):
        return 'nvidia'
    if any(keyword in text for keyword in ('amd', 'radeon', 'ati', 'rx ', 'instinct')):
        return 'amd'
    if any(keyword in text for keyword in ('intel', 'arc', 'iris', 'uhd', 'integrated')):
        return 'intel'
    return 'unknown'


def normalize_gpu_record(raw):
    """Normalize GPU telemetry into a consistent schema for the frontend."""
    util = safe_float(raw.get('utilization'), 0.0)
    avg_util = raw.get('utilization_avg')
    if avg_util is None:
        avg_util = util

    return {
        'name': raw.get('name') or 'GPU',
        'utilization': round(util, 1),
        'utilization_avg': round(safe_float(avg_util, util), 1),
        'temperature_c': raw.get('temperature_c') if raw.get('temperature_c') is not None else 0,
        'memory_used_mb': round(safe_float(raw.get('memory_used_mb'), 0.0), 1),
        'memory_total_mb': round(safe_float(raw.get('memory_total_mb'), 0.0), 1),
        'memory_used_pct': round(safe_float(raw.get('memory_used_pct'), 0.0), 1),
        'clock_mhz': round(safe_float(raw.get('clock_mhz'), 0.0), 1),
        'vendor': raw.get('vendor') or detect_vendor(raw.get('name')),
        'source': raw.get('source') or 'unknown'
    }


def build_gpu_status(detected, reason=''):
    """Create a structured status payload for the GPU detection pipeline."""
    return {
        'detected': bool(detected),
        'reason': reason or ('GPU detected successfully' if detected else 'GPU detection failed')
    }


def collect_gpu_samples(limit=6):
    """Collect a small set of GPU samples for diagnostics and benchmarking."""
    samples = []
    for _ in range(limit):
        gpu_data, _ = get_gpu_stats()
        if gpu_data:
            first = gpu_data[0]
            samples.append({
                'utilization': first.get('utilization', 0),
                'temperature_c': first.get('temperature_c'),
                'memory_used_mb': first.get('memory_used_mb', 0)
            })
        time.sleep(0.5)
    return samples


def run_gpu_diagnostics():
    """Run a lightweight health check and return a detailed GPU status report."""
    gpus, detection_status = get_gpu_stats()
    detected = bool(gpus)

    driver_installed = detection_status.get('detected', False)
    nvml_working = bool(NVML_AVAILABLE and pynvml is not None and detected and any(g.get('source') == 'pynvml' for g in gpus))
    vram_accessible = any(
        g.get('memory_total_mb') not in (None, 0) and g.get('memory_used_mb') is not None
        for g in gpus
    )

    utilization_updating = False
    temperature_updating = False
    for idx, gpu in enumerate(gpus):
        values = GPU_UTIL_HISTORY.get(idx, deque(maxlen=10))
        if len(values) >= 2:
            utilization_updating = utilization_updating or (max(values) - min(values) >= 1)
        if gpu.get('temperature_c') not in (None, 0):
            temperature_updating = temperature_updating or (gpu.get('temperature_c', 0) > 0)

    report = {
        'detected': detected,
        'driver_installed': driver_installed,
        'nvml_working': nvml_working,
        'vram_accessible': vram_accessible,
        'utilization_updating': utilization_updating,
        'temperature_updating': temperature_updating,
        'source': gpus[0].get('source') if gpus else None,
        'vendor': gpus[0].get('vendor') if gpus else None,
        'reason': detection_status.get('reason') if detection_status else 'GPU diagnostic unavailable',
        'summary': 'GPU healthy' if detected and all([
            driver_installed,
            nvml_working or vram_accessible,
            utilization_updating or detected,
            temperature_updating or not detected
        ]) else 'GPU health check failed'
    }
    report['all_clear'] = bool(
        report['detected'] and
        report['driver_installed'] and
        report['vram_accessible'] and
        report['utilization_updating'] and
        report['temperature_updating']
    )
    return report


def run_gpu_stress_test(duration=3):
    """Probe GPU activity over a short interval to confirm metrics are updating."""
    duration = max(1, int(duration))
    start = time.monotonic()
    samples = []

    while (time.monotonic() - start) < duration:
        gpu_data, _ = get_gpu_stats()
        if gpu_data:
            first = gpu_data[0]
            samples.append({
                'utilization': first.get('utilization', 0),
                'temperature_c': first.get('temperature_c'),
                'memory_used_mb': first.get('memory_used_mb', 0)
            })
        time.sleep(0.5)

    if not samples:
        return {
            'status': 'warning',
            'duration_sec': duration,
            'message': 'No GPU samples collected during stress test'
        }

    util_values = [s['utilization'] for s in samples]
    temp_values = [s['temperature_c'] for s in samples if s['temperature_c'] is not None]
    change = max(util_values) - min(util_values) if util_values else 0
    return {
        'status': 'ok' if change >= 1 else 'warning',
        'duration_sec': duration,
        'samples': len(samples),
        'utilization_delta': round(change, 1),
        'avg_utilization': round(sum(util_values) / len(util_values), 1) if util_values else 0,
        'avg_temperature': round(sum(temp_values) / len(temp_values), 1) if temp_values else None,
        'message': 'GPU metrics updated during stress probe' if change >= 1 else 'GPU metrics stayed flat during stress probe'
    }


def run_gpu_benchmark(duration=3):
    """Benchmark GPU telemetry responsiveness over a short observation window."""
    duration = max(1, int(duration))
    start = time.monotonic()
    samples = []
    while (time.monotonic() - start) < duration:
        gpu_data, _ = get_gpu_stats()
        if gpu_data:
            first = gpu_data[0]
            samples.append({
                'utilization': first.get('utilization', 0),
                'temperature_c': first.get('temperature_c'),
                'memory_used_mb': first.get('memory_used_mb', 0)
            })
        time.sleep(0.5)

    if not samples:
        return {
            'status': 'warning',
            'duration_sec': duration,
            'message': 'No GPU samples collected during benchmark'
        }

    util_values = [s['utilization'] for s in samples]
    temp_values = [s['temperature_c'] for s in samples if s['temperature_c'] is not None]
    return {
        'status': 'ok',
        'duration_sec': duration,
        'sample_count': len(samples),
        'avg_utilization': round(sum(util_values) / len(util_values), 1) if util_values else 0,
        'max_utilization': round(max(util_values), 1) if util_values else 0,
        'avg_temperature': round(sum(temp_values) / len(temp_values), 1) if temp_values else None,
        'max_temperature': round(max(temp_values), 1) if temp_values else None,
        'message': 'Benchmark completed successfully'
    }


def run_command(command, timeout=2):
    """Run a subprocess command safely and return the result object."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


class RealFpsMonitor:
    """Estimate FPS from real screen updates using dxcam on Windows."""

    def __init__(self):
        self.capture = None
        self._last_timestamp = None
        self._history = deque(maxlen=8)
        self._last_fps = 0.0
        self._error = None

    def start(self):
        if platform.system() != 'Windows' or dxcam is None:
            self._error = 'FPS detection requires Windows and the dxcam dependency.'
            return False

        try:
            self.capture = dxcam.create(output_idx=0)
            if not self.capture.is_capturing:
                self.capture.start(target_fps=60)
            self._last_timestamp = None
            self._error = None
            return True
        except Exception as exc:  # pragma: no cover - runtime specific
            self._error = str(exc)
            self.capture = None
            return False

    def sample(self):
        """Return a real FPS estimate and a human-readable status."""
        if self.capture is None:
            started = self.start()
            if not started:
                return {
                    'fps': 0,
                    'status': 'unavailable',
                    'message': 'Real FPS is unavailable on this machine. Install dxcam and opencv-python on Windows, then run a supported display capture session.',
                    'source': 'dxcam-unavailable'
                }

        try:
            latest = self.capture.get_latest_frame(with_timestamp=True)
            if latest is None:
                return {
                    'fps': 0,
                    'status': 'waiting',
                    'message': 'FPS capture is active but no fresh frame has been received yet.',
                    'source': 'dxcam'
                }

            frame, frame_timestamp = latest
            if frame is None:
                return {
                    'fps': 0,
                    'status': 'waiting',
                    'message': 'FPS capture is active but no fresh frame has been received yet.',
                    'source': 'dxcam'
                }

            if self._last_timestamp is not None and frame_timestamp > self._last_timestamp:
                elapsed = frame_timestamp - self._last_timestamp
                if elapsed > 0:
                    fps = 1.0 / elapsed
                    self._history.append(fps)
                    self._last_fps = round(sum(self._history) / len(self._history), 1)

            self._last_timestamp = frame_timestamp

            return {
                'fps': self._last_fps if self._last_fps > 0 else 0,
                'status': 'live' if self._last_fps > 0 else 'waiting',
                'message': 'Real FPS measured from live screen capture.',
                'source': 'dxcam'
            }
        except Exception as exc:  # pragma: no cover - runtime specific
            self._error = str(exc)
            return {
                'fps': 0,
                'status': 'unavailable',
                'message': f'FPS capture failed: {exc}',
                'source': 'dxcam-error'
            }


FPS_MONITOR = RealFpsMonitor()


# ═══ HELPER: PC ka data collect karo ═══
def get_system_stats():
    """
    psutil se saara system data collect karta hai.
    Ek dictionary return karta hai jo JSON mein convert hogi.
    """

    fps_info = FPS_MONITOR.sample()

    # CPU — interval=0.5 means 0.5s average (zyada accurate)
    cpu_pct = psutil.cpu_percent(interval=0.5)
    cpu_freq = psutil.cpu_freq()

    # RAM
    mem = psutil.virtual_memory()

    # Disk — Windows pe 'C:\\', Linux/Mac pe '/'
    try:
        disk = psutil.disk_usage('C:\\')   # Windows
    except:
        disk = psutil.disk_usage('/')      # Linux/Mac

    # Network — per-second bytes (speed calculate karne ke liye)
    net = psutil.net_io_counters()

    # GPU — collect info via NVML or nvidia-smi as fallback
    gpus, gpu_status = get_gpu_stats()

    stats = {
        # ── CPU ──
        'cpu':          round(cpu_pct, 1),
        'cpu_cores':    psutil.cpu_count(),
        'cpu_freq_mhz': round(cpu_freq.current, 0) if cpu_freq else 0,

        # ── RAM ──
        'ram':          round(mem.percent, 1),
        'ram_used_gb':  round(mem.used  / 1024**3, 2),
        'ram_total_gb': round(mem.total / 1024**3, 2),

        # ── DISK ──
        'disk':         round(disk.percent, 1),
        'disk_used_gb': round(disk.used  / 1024**3, 1),
        'disk_total_gb':round(disk.total / 1024**3, 1),

        # ── FPS (real or unavailable) ──
        'fps':          fps_info['fps'],
        'fps_status':   fps_info['status'],
        'fps_source':   fps_info['source'],
        'fps_message':  fps_info['message'],

        # ── Metadata ──
        'timestamp':    int(time.time()),
        'status':       'live' if fps_info['status'] in ('live', 'waiting') else 'warning'
    }

    stats['gpus'] = gpus
    stats['gpu'] = gpus[0] if gpus else None
    stats['gpu_status'] = gpu_status
    return stats


def get_fps_status():
    """Expose the FPS monitor status for tests and debugging."""
    return FPS_MONITOR.sample()


# ═══ GPU HELPERS ═══
def parse_nvidia_smi_output(output):
    """Parse CSV-style output from nvidia-smi --query-gpu and return list of dicts."""
    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        name, util, temp, mem_total, mem_used, clock = parts[:6]
        try:
            util_f = float(util)
        except Exception:
            util_f = 0.0
        try:
            temp_f = float(temp)
        except Exception:
            temp_f = None
        try:
            mem_total_mb = float(mem_total)
            mem_used_mb = float(mem_used)
            mem_pct = round((mem_used_mb / mem_total_mb) * 100, 1) if mem_total_mb else 0
        except Exception:
            mem_total_mb = mem_used_mb = mem_pct = None
        try:
            clock_mhz = float(clock)
        except Exception:
            clock_mhz = None

        gpus.append({
            'name': name,
            'utilization': round(util_f, 1),
            'temperature_c': temp_f,
            'memory_total_mb': mem_total_mb,
            'memory_used_mb': mem_used_mb,
            'memory_used_pct': mem_pct,
            'clock_mhz': clock_mhz,
            'source': 'nvidia-smi'
        })
    return gpus


def get_nvidia_gpu_stats():
    """Collect NVIDIA GPU stats using NVML first, then nvidia-smi."""
    if NVML_AVAILABLE and pynvml is not None:
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return [], build_gpu_status(False, 'No NVIDIA GPUs detected by NVML')
            result = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                name = name.decode('utf-8') if isinstance(name, bytes) else name
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    util = 0
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_total_mb = round(mem.total / 1024**2, 1)
                    mem_used_mb = round(mem.used / 1024**2, 1)
                    mem_pct = round(mem.used / mem.total * 100, 1) if mem.total else None
                except Exception:
                    mem_total_mb = mem_used_mb = mem_pct = None
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                try:
                    clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                except Exception:
                    clock = None

                history = GPU_UTIL_HISTORY.setdefault(i, deque(maxlen=10))
                if util is not None:
                    history.append(float(util))
                avg_util = compute_rolling_average(history, window=10)

                result.append({
                    'name': name,
                    'utilization': round(util, 1) if util is not None else 0,
                    'utilization_avg': avg_util,
                    'temperature_c': temp,
                    'memory_total_mb': mem_total_mb,
                    'memory_used_mb': mem_used_mb,
                    'memory_used_pct': mem_pct,
                    'clock_mhz': clock,
                    'vendor': 'nvidia',
                    'source': 'pynvml'
                })
            return [normalize_gpu_record(gpu) for gpu in result], build_gpu_status(True, 'NVIDIA GPU detected via NVML')
        except Exception as exc:
            return [], build_gpu_status(False, f'NVML error: {exc}')

    try:
        cmd = 'nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,memory.total,memory.used,clocks.current.graphics --format=csv,noheader,nounits'
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=2)
        if proc.returncode == 0 and proc.stdout:
            parsed = parse_nvidia_smi_output(proc.stdout)
            normalized = []
            for idx, gpu in enumerate(parsed):
                history = GPU_UTIL_HISTORY.setdefault(idx, deque(maxlen=10))
                util = gpu.get('utilization')
                if util is not None:
                    history.append(float(util))
                gpu['utilization_avg'] = compute_rolling_average(history, window=10)
                gpu['vendor'] = 'nvidia'
                normalized.append(normalize_gpu_record(gpu))
            return normalized, build_gpu_status(True, 'NVIDIA GPU detected via nvidia-smi')
        elif proc.returncode != 0:
            stderr = proc.stderr.strip() if proc.stderr else ''
            stdout = proc.stdout.strip() if proc.stdout else ''
            msg = stderr or stdout or 'nvidia-smi command failed'
            return [], build_gpu_status(False, f'nvidia-smi error: {msg}')
    except FileNotFoundError:
        return [], build_gpu_status(False, 'nvidia-smi not found')
    except PermissionError:
        return [], build_gpu_status(False, 'permission denied while running nvidia-smi')
    except Exception as exc:
        return [], build_gpu_status(False, f'nvidia-smi unexpected error: {exc}')

    return [], build_gpu_status(False, 'no NVIDIA GPU detected or NVIDIA tools unavailable')


def parse_amd_output(output):
    """Try to parse AMD telemetry from common ROCm/AMD CLI outputs."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    name = None
    util = None
    temp = None
    mem_used = None
    mem_total = None
    clock = None

    for line in lines:
        lowered = line.lower()
        if name is None and any(keyword in lowered for keyword in ('product name', 'name:', 'gpu name')):
            parts = line.split(':', 1)
            if len(parts) > 1:
                name = parts[1].strip()
        if util is None:
            maybe = None
            if 'gpu use' in lowered or 'gpu usage' in lowered or 'gpu util' in lowered or 'utilization' in lowered:
                for part in line.split():
                    if '%' in part:
                        maybe = part.replace('%', '')
                        break
                if maybe is not None:
                    try:
                        util = float(maybe)
                    except Exception:
                        util = None
        if temp is None and any(keyword in lowered for keyword in ('temp', 'temperature')):
            for part in line.split():
                if part.replace('.', '', 1).isdigit():
                    temp = float(part)
                    break
        if mem_used is None and any(keyword in lowered for keyword in ('vram used', 'mem used', 'memory used')):
            for part in line.split():
                if part.replace('.', '', 1).isdigit():
                    mem_used = float(part)
                    break
        if mem_total is None and any(keyword in lowered for keyword in ('vram total', 'mem total', 'memory total')):
            for part in line.split():
                if part.replace('.', '', 1).isdigit():
                    mem_total = float(part)
                    break
        if clock is None and any(keyword in lowered for keyword in ('clock', 'frequency')):
            for part in line.split():
                if part.replace('.', '', 1).isdigit():
                    clock = float(part)
                    break

    return [{
        'name': name or 'AMD GPU',
        'utilization': util if util is not None else 0,
        'temperature_c': temp,
        'memory_used_mb': mem_used,
        'memory_total_mb': mem_total,
        'memory_used_pct': round((mem_used / mem_total) * 100, 1) if mem_used and mem_total else 0,
        'clock_mhz': clock,
        'source': 'amd-cli'
    }]


def get_amd_gpu_stats():
    """Try common AMD-specific tools and return normalized GPU data when available."""
    commands = [
        ['rocm-smi', '--showproductname', '--showuse', '--showtemp', '--showmeminfo'],
        ['amd-smi', 'list'],
        ['amd-smi', '--help']
    ]

    for cmd in commands:
        proc = run_command(cmd, timeout=2)
        if proc and proc.returncode == 0 and proc.stdout:
            parsed = parse_amd_output(proc.stdout)
            if parsed:
                normalized = []
                for idx, gpu in enumerate(parsed):
                    history = GPU_UTIL_HISTORY.setdefault(idx, deque(maxlen=10))
                    util = safe_float(gpu.get('utilization'), 0.0)
                    if util is not None:
                        history.append(float(util))
                    gpu['utilization_avg'] = compute_rolling_average(history, window=10)
                    gpu['vendor'] = 'amd'
                    gpu['name'] = gpu.get('name') or 'AMD GPU'
                    normalized.append(normalize_gpu_record(gpu))
                return normalized, build_gpu_status(True, f'AMD GPU detected via {cmd[0]}')

    return [], build_gpu_status(False, 'unsupported AMD GPU or AMD tools unavailable')


def get_intel_gpu_stats():
    """Try Intel-specific tools and return normalized GPU data when available."""
    commands = [
        ['intel_gpu_top', '-J'],
        ['intel_gpu_top', '-o', 'json']
    ]

    for cmd in commands:
        proc = run_command(cmd, timeout=2)
        if proc and proc.returncode == 0 and proc.stdout:
            try:
                import json as json_lib
                data = json_lib.loads(proc.stdout)
                if isinstance(data, dict):
                    name = data.get('name') or data.get('device') or 'Intel GPU'
                    util = data.get('utilization') or data.get('gpu_util') or 0
                    temp = data.get('temperature') or data.get('temp')
                    mem_used = data.get('memory_used') or data.get('used_memory')
                    mem_total = data.get('memory_total') or data.get('total_memory')
                    parsed = {
                        'name': name,
                        'utilization': util,
                        'temperature_c': temp,
                        'memory_used_mb': mem_used,
                        'memory_total_mb': mem_total,
                        'memory_used_pct': round((mem_used / mem_total) * 100, 1) if mem_used and mem_total else 0,
                        'clock_mhz': data.get('clock') or data.get('frequency') or 0,
                        'vendor': 'intel',
                        'source': 'intel-cli'
                    }
                    history = GPU_UTIL_HISTORY.setdefault(0, deque(maxlen=10))
                    util_value = safe_float(parsed.get('utilization'), 0.0)
                    if util_value is not None:
                        history.append(float(util_value))
                    parsed['utilization_avg'] = compute_rolling_average(history, window=10)
                    return [normalize_gpu_record(parsed)], build_gpu_status(True, f'Intel GPU detected via {cmd[0]}')
            except Exception:
                pass

    return [], build_gpu_status(False, 'unsupported Intel GPU or Intel tools unavailable')


def get_gpu_stats():
    """Return a unified list of GPU dictionaries for NVIDIA, AMD, and Intel GPUs."""
    gpu_results, status = get_nvidia_gpu_stats()
    if gpu_results:
        return gpu_results, status

    gpu_results, status = get_amd_gpu_stats()
    if gpu_results:
        return gpu_results, status

    gpu_results, status = get_intel_gpu_stats()
    if gpu_results:
        return gpu_results, status

    return [], build_gpu_status(False, 'No supported GPU detected. Check drivers, permissions, or vendor tools.')


# ═══ REST API ENDPOINT ═══
# Browser ek baar data maang sakta hai (polling ke liye)
@app.route('/api/stats')
def api_stats():
    """Single snapshot — HTTP GET /api/stats"""
    return jsonify(get_system_stats())


@app.route('/api/gpu/diagnostics')
def gpu_diagnostics():
    """Return a detailed GPU health report."""
    return jsonify({
        'timestamp': int(time.time()),
        'diagnostics': run_gpu_diagnostics()
    })


@app.route('/api/gpu/stress-test')
def gpu_stress_test():
    """Run a short GPU telemetry stress probe."""
    return jsonify({
        'timestamp': int(time.time()),
        'stress_test': run_gpu_stress_test()
    })


@app.route('/api/gpu/benchmark')
def gpu_benchmark():
    """Run a short GPU telemetry benchmark."""
    return jsonify({
        'timestamp': int(time.time()),
        'benchmark': run_gpu_benchmark()
    })


# ═══ HEALTH CHECK ═══
@app.route('/')
def health():
    return jsonify({
        'status': 'running',
        'message': 'System Monitor Backend chal raha hai! ✅',
        'endpoints': ['/api/stats', '/ws']
    })


# ═══ WEBSOCKET ENDPOINT ═══
# Browser yahan permanent connection banata hai
# Server automatically har 1 second mein data bhejta hai
@sock.route('/ws')
def websocket_stream(ws):
    """
    Real-time WebSocket stream.
    Har 1 second mein PC stats JSON format mein bhejta hai.
    Connection todne par loop automatically band ho jaata hai.
    """
    print("\n✅ Browser connected via WebSocket!")
    print("   Data stream starting...")

    while True:
        try:
            # Asli PC data padhna
            stats = get_system_stats()

            # JSON string mein convert karo
            payload = json.dumps(stats)

            # Browser ko bhejo!
            ws.send(payload)

            # Debug: terminal pe bhi print karo
            print(f"  📡 Sent: CPU={stats['cpu']}%  RAM={stats['ram']}%  FPS={stats['fps']}", end='\r')

            # 1 second ruko
            time.sleep(1)

        except Exception as e:
            print(f"\n❌ Connection lost: {e}")
            break

    print("\n🔌 Browser disconnected. Waiting for next connection...")


# ═══ SERVER START ═══
if __name__ == '__main__':
    print("""
╔════════════════════════════════════════╗
║   System Monitor Backend Starting...   ║
╠════════════════════════════════════════╣
║  REST API: http://localhost:5000/api/stats
║  WebSocket: ws://localhost:5000/ws
║  Health:   http://localhost:5000/
╚════════════════════════════════════════╝
Press Ctrl+C to stop.
""")
    # use_reloader=False — WebSocket ke saath better stability
    app.run(
    host='0.0.0.0',
    port=int(os.environ.get("PORT", 5000)),
    debug=False,
    use_reloader=False
)