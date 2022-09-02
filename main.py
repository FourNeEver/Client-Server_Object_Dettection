# echo-server
import socket

HOST = "192.168.1.74"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            data.decode('utf-8')
            f = open("test.txt.", "w")
            text = data.decode('utf-8')
            f.write(text)
            print(text)
            f.close()
            if not data:
                break
            conn.sendall(data)