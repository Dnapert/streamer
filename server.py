import socket,cv2,pickle,struct,imutils
from ultralytics import YOLO
import threading
import uuid
import signal
import torch
model = YOLO('yolov8n.pt')

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")
    
stop_signal = False

def handle_signal(sig,frame):
    global stop_signal
    stop_signal = True
    print("Stopping server gracefully...")
    exit(0)
    
# Register signal handlers
signal.signal(signal.SIGINT,handle_signal)
signal.signal(signal.SIGTERM,handle_signal)


def create_socket():
    try:
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        host_name  = socket.gethostname()
        host_ip = socket.gethostbyname(host_name) 
        print('HOST IP:',host_ip)
        port = 6942
        socket_address = (host_ip,port)
        server_socket.bind(socket_address)
        server_socket.listen(5)
    except socket.error as msg:
        print("Socket creation error: "+str(msg))
        server_socket = None
    return server_socket,host_ip
  
def get_cap():
    vid = cv2.VideoCapture(0)
    return vid

# this is what will run in a thread
def handle_client(client_socket,addr,vid,client_id):
    print('GOT CONNECTION FROM:',addr)
    connack_msg = f"CONNACK {client_id}"
    client_socket.send(connack_msg.encode())
    try:
        while(vid.isOpened()):
            img,frame = vid.read()
            frame = imutils.resize(frame,width=640)
            if img is not None:
                results = model.track(frame,conf=.5,verbose=False,device=device,save=False,show=False)
                frame = results[0].plot()
            a = pickle.dumps(frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)
    except Exception as e:
        print(f"Disconnected from client {client_id}")    
    finally:
        client_socket.close()

vid = get_cap()
server_socket,host_ip = create_socket()

if server_socket is None:
    print("Error creating socket, program exiting...")
    vid.release()
    exit(0)
    
while True:
    client_socket,addr = server_socket.accept()
    client_id = str(uuid.uuid4()).split('-')[0]
    client_count = threading.active_count() - 1
    if client_count == 0:
        print("STARTING VIDEO STREAM")
    if client_count > 3:
        print("MAXIMUM CLIENTS REACHED")
        continue
    thread = threading.Thread(target=handle_client,args=(client_socket,addr,vid,client_id))
    thread.start()
    print("TOTAL CLIENTS ",threading.active_count() - 1)
    if stop_signal:
        break
print("STOPPING VIDEO STREAM")
vid.release()
cv2.destroyAllWindows()
server_socket.close()