"""An owned local Flower service bound to the active Python environment."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen


def available_ports():
    sockets = [socket.socket(), socket.socket()]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


@contextmanager
def local_service(env, log_path):
    """Never reuse or stop another project's managed SuperLink."""
    http_port, control_port = available_ports()
    with TemporaryDirectory(prefix="damldg-flower-state-") as state:
        child_env = dict(env)
        child_env.update(
            FLWR_HOME=state,
            FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=str(http_port),
            FLWR_LOCAL_CONTROL_API_PORT=str(control_port),
            FLWR_DISABLE_UPDATE_CHECK="1",
        )
        command = [
            sys.executable,
            "-c",
            "from flwr.superlink.cli import flower_superlink; flower_superlink()",
            "--insecure",
            "--simulation",
            "--isolation",
            "subprocess",
            "--disable-runtime-dependency-installation",
            "--control-api-address",
            f"127.0.0.1:{control_port}",
            "--host",
            "127.0.0.1",
            "--port",
            str(http_port),
        ]
        with Path(log_path).open("w") as log:
            process = subprocess.Popen(
                command,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Flower service exited; inspect {log_path}")
                    try:
                        with urlopen(
                            f"http://127.0.0.1:{http_port}/health", timeout=1
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError(
                        f"Flower service did not start; inspect {log_path}"
                    )
                yield child_env
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
