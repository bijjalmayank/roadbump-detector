import os
import signal
import subprocess

def kill_process_on_port(port=5000):
    try:
        # Find process listening on the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pids = result.stdout.strip().split('\n')

        if not pids or pids == ['']:
            print(f"No process is running on port {port}")
            return

        for pid in pids:
            print(f"Killing process {pid} on port {port}")
            os.kill(int(pid), signal.SIGKILL)

        print(f"Successfully killed processes on port {port}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    kill_process_on_port(5000)
