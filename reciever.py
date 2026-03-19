import socket
import os

SAVE_DIR = "ReceivedFromPi"
CHANNEL  = 12
os.makedirs(SAVE_DIR, exist_ok=True)

server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
server_sock.bind(("70:D8:23:96:BA:DA", CHANNEL))
server_sock.listen(1)
print("Waiting for Pi to connect...")

while True:
    client_sock, address = server_sock.accept()
    print(f"Connected: {address}")
    try:
        header = b""
        while b"\n" not in header:
            header += client_sock.recv(1)
        filename, size = header.decode().strip().split(":")
        size = int(size)
        data = b""
        while len(data) < size:
            chunk = client_sock.recv(4096)
            if not chunk:
                break
            data += chunk
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"Saved {filename} ({size} bytes)")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_sock.close()