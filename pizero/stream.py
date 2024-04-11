import socket,cv2,pickle,struct
from picamera2 import Picamera2
cam = Picamera2()
cam.start()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 6942))
server_socket.listen(5)
print("Server is listening...")
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address} accepted")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    image = cam.capture_array()
    frame_data = pickle.dumps(image)
    client_socket.sendall(struct.pack("Q", len(frame_data)))
    client_socket.sendall(frame_data)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()