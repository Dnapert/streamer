import socket,struct, pickle,cv2, signal

stop_signal = False

def handle_signal(sig,frame):
    global stop_signal
    stop_signal = True
    print("Stopping client gracefully...")
    exit(0) 
# Register signal handlers
signal.signal(signal.SIGINT,handle_signal)
signal.signal(signal.SIGTERM,handle_signal)

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '127.0.1.1'
port = 6942
socket_address = (host_ip,port)
client_socket.connect(socket_address)
connack_msg = client_socket.recv(1024).decode()
client_id = connack_msg.split(' ')[1]
print(connack_msg)
data = b""
payload_size = struct.calcsize("Q")
print(payload_size)
while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet:
            break
        data+=packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q",packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame,data = data[:msg_size],data[msg_size:]
    
    frame = pickle.loads(frame)
    cv2.imshow(f"Client {client_id}",frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    if stop_signal:
        break
cv2.destroyAllWindows()
client_socket.close()