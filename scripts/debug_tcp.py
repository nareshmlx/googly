
import socket
import time

host = "13.228.184.177"
port = 5432

print(f"--- TESTING TCP CONNECTION TO {host}:{port} ---")
start = time.time()
try:
    s = socket.create_connection((host, port), timeout=5)
    print("Connection SUCCESS!")
    s.close()
except Exception as e:
    print(f"Connection FAILED: {e}")
print(f"Time: {time.time() - start:.2f}s")
