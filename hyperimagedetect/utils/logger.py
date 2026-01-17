
import logging
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from hyperimagedetect.utils import MACOS, RANK
from hyperimagedetect.utils.checks import check_requirements

DEFAULT_LOG_PATH = Path("train.log")
if RANK in {-1, 0} and DEFAULT_LOG_PATH.exists():
    DEFAULT_LOG_PATH.unlink(missing_ok=True)

class ConsoleLogger:

    def __init__(self, destination):
        self.destination = Path(destination)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False

    def start_capture(self):
        if self.active:
            return

        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

        try:
            handler = self._LogHandler(self._queue_log)
            logging.getLogger("hyperimagedetect").addHandler(handler)
        except Exception:
            pass

        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def stop_capture(self):
        if not self.active:
            return

        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_queue.put(None)

    def _queue_log(self, text):
        if not self.active:
            return

        current_time = time.time()

        if "\r" in text:
            text = text.split("\r")[-1]

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        for line in lines:
            line = line.rstrip()

            if "─" in line:
                continue

            if " ━━" in line:
                progress_core = line.split(" ━━")[0].strip()
                if progress_core == self.last_progress_line and self.last_was_progress:
                    continue
                self.last_progress_line = progress_core
                self.last_was_progress = True
            else:

                if not line and self.last_was_progress:
                    self.last_was_progress = False
                    continue
                self.last_was_progress = False

            if line == self.last_line and current_time - self.last_time < 0.1:
                continue

            self.last_line = line
            self.last_time = current_time

            if not line.startswith("[20"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"

            if not self._safe_put(f"{line}\n"):
                continue

    def _safe_put(self, item):
        try:
            self.log_queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(item)
                return True
            except queue.Empty:
                return False

    def _stream_worker(self):
        while self.active:
            try:
                log_text = self.log_queue.get(timeout=1)
                if log_text is None:
                    break
                self._write_log(log_text)
            except queue.Empty:
                continue

    def _write_log(self, text):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as f:
            f.write(text)

    class _ConsoleCapture:

        __slots__ = ("callback", "original")

        def __init__(self, original, callback):
            self.original = original
            self.callback = callback

        def write(self, text):
            self.original.write(text)
            self.callback(text)

        def flush(self):
            self.original.flush()

    class _LogHandler(logging.Handler):

        __slots__ = ("callback",)

        def __init__(self, callback):
            super().__init__()
            self.callback = callback

        def emit(self, record):
            self.callback(self.format(record) + "\n")

class SystemLogger:

    def __init__(self):
        import psutil

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()

        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        try:
            assert not MACOS
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def get_metrics(self, rates=False):
        import psutil

        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        disk_usage = shutil.disk_usage("/")
        now = time.time()

        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "gpus": {},
        }

        elapsed = max(0.1, now - self._prev_time)

        if rates:

            metrics["disk"] = {
                "read_mbs": round(max(0, (disk.read_bytes - self._prev_disk.read_bytes) / (1 << 20) / elapsed), 3),
                "write_mbs": round(max(0, (disk.write_bytes - self._prev_disk.write_bytes) / (1 << 20) / elapsed), 3),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / (1 << 20) / elapsed), 3),
                "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / (1 << 20) / elapsed), 3),
            }
        else:

            metrics["disk"] = {
                "read_mb": round((disk.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3),
                "write_mb": round((disk.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3),
                "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3),
            }

        self._prev_net = net
        self._prev_disk = disk
        self._prev_time = now

        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())

        return metrics

    def _get_nvidia_metrics(self):
        gpus = {}
        if not self.nvidia_initialized or not self.pynvml:
            return gpus
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000

                gpus[str(i)] = {
                    "usage": round(util.gpu, 3),
                    "memory": round((memory.used / memory.total) * 100, 3),
                    "temp": temp,
                    "power": power,
                }
        except Exception:
            pass
        return gpus

if __name__ == "__main__":
    print("SystemLogger Real-time Metrics Monitor")
    print("Press Ctrl+C to stop\n")

    logger = SystemLogger()

    try:
        while True:
            metrics = logger.get_metrics()

            print("\033[H\033[J", end="")

            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"Disk Read: {metrics['disk']['read_mb']:8.1f} MB")
            print(f"Disk Write: {metrics['disk']['write_mb']:7.1f} MB")
            print(f"Disk Used: {metrics['disk']['used_gb']:8.1f} GB")
            print(f"Net Recv: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"Net Sent: {metrics['network']['sent_mb']:9.1f} MB")

            if metrics["gpus"]:
                print("\nGPU Metrics:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | "
                        f"Mem: {gpu_data['memory']:5.1f}% | "
                        f"Temp: {gpu_data['temp']:2}°C | "
                        f"Power: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: No NVIDIA GPUs detected")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")