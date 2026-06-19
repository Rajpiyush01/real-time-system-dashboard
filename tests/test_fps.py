import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'server.py'

spec = importlib.util.spec_from_file_location('server_module', MODULE_PATH)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_fps_report_includes_status_and_source():
    stats = server.get_system_stats()

    assert 'fps' in stats
    assert 'fps_status' in stats
    assert 'fps_source' in stats
    assert isinstance(stats['fps_status'], str)
    assert isinstance(stats['fps_source'], str)


def test_fps_helper_returns_meaningful_value_when_unavailable():
    result = server.get_fps_status()

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'message' in result


def test_gpu_status_report_includes_detection_info():
    stats = server.get_system_stats()

    assert 'gpu_status' in stats
    assert isinstance(stats['gpu_status'], dict)
    assert 'detected' in stats['gpu_status']
    assert 'reason' in stats['gpu_status']


def test_compute_rolling_average_uses_recent_samples():
    assert server.compute_rolling_average([0, 0, 40, 80], window=3) == 40.0
