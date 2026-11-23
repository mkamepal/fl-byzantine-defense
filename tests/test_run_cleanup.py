"""Exercise launcher cleanup with real subprocesses, without Flower installed."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from run import cleanup_process_group, run_command


def running(pid):
    try:
        # An orphan zombie is stopped and awaits reaping by the system.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux process inspection")
class CleanupTests(unittest.TestCase):
    def test_cleanup_after_success_and_failure(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                pid_file = Path(tmp) / "pid"
                code = (
                    "import subprocess,sys; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-S','-c',"
                    "'import time; time.sleep(60)']); "
                    f"Path({str(pid_file)!r}).write_text(str(p.pid)); "
                    f"sys.exit({exit_code})"
                )
                self.assertEqual(run_command([sys.executable, "-S", "-c", code]), exit_code)
                self.assertFalse(running(int(pid_file.read_text())))

    def test_forces_unresponsive_process_to_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            code = (
                "import signal,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(ready)!r}).touch(); time.sleep(60)"
            )
            process = subprocess.Popen([sys.executable, "-S", "-c", code], start_new_session=True)
            try:
                self.wait_for_file(ready)
                cleanup_process_group(process, timeout=0.2)
                self.assertEqual(process.returncode, -signal.SIGKILL)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def test_interrupts_clean_up_workers(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as tmp:
                pid_file = Path(tmp) / "pid"
                worker = (
                    "import os,time; from pathlib import Path; "
                    f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
                )
                wrapper = (
                    "from run import run_command; import sys; "
                    f"run_command([sys.executable, '-S', '-c', {worker!r}])"
                )
                process = subprocess.Popen([sys.executable, "-S", "-c", wrapper])
                try:
                    self.wait_for_file(pid_file)
                    process.send_signal(signum)
                    self.assertEqual(process.wait(timeout=10), 128 + signum)
                    self.assertFalse(running(int(pid_file.read_text())))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)

    def wait_for_file(self, path):
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(path.exists(), "Child did not become ready")


if __name__ == "__main__":
    unittest.main()
