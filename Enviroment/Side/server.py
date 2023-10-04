import socket
import threading
import json
              
class Server:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.receive_size = 1024 * 1024 * 10
        self.control_proccess = threading.Thread(target=self.control_server_receive)
        self.receive_event = threading.Event()
        self.create_server()
        self.accept_client()

    def accept_client(self, client_count=2):
        self.server_socket.listen(client_count)
        print("Listening for connections...")
        self.client_socket, addr = self.server_socket.accept()
        print(f"New CLİENT socket connected: {addr[0]}:{addr[1]}")
        self.control_socket, addr = self.server_socket.accept()
        print(f"New CONTROL socket connected: {addr[0]}:{addr[1]}")
        print("Client is ready.")

    def create_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_ip, self.server_port))
        print("Created socket at", self.server_ip, "on port", self.server_port, "...")

    def receive_from_client(self) -> list[list]:
        self.receive_event.set()
        self.control_proccess.join()
        self.control_proccess = threading.Thread(target=self.control_server_receive)
        print("Server waiting for data(receiving)...")
        received_data = self.client_socket.recv(self.receive_size).decode('utf-8')
        try:
            received_data = json.loads(received_data)
        except Exception as e:  
            print(f"JSON Error: {e}")
        # print("Server received data... Bone Transform:", received_data[0])
        self.receive_event.clear()
        return received_data

    def send_to_client(self, data, control_receive=True):
        self.control_client_receive()
        # print("Server sending data...", data)
        try:
            data = json.dumps(data).encode('utf-8')
        except Exception as e:
            print(f"JSON Error: {e}")
        self.client_socket.sendall(data)
        if control_receive:
            self.control_proccess.start()
        print("Server successfully sent data.")

    def control_client_receive(self):
        try:
            self.control_socket.sendall("IS_READY".encode("utf-8"))
            print("Server sended IS_READY and Server waiting for client ready...")
            self.control_socket.settimeout(3)
            self.control_socket.recv(1024).decode("utf-8")
            self.control_socket.settimeout(None)
            print("Client said 'i am ready to receive.'")  
        except socket.error as e:
            print(f"Control Socket Error and disconnect: {e}")
            self.disconnect(keep_server=False)
    
    #IS_READY aldıktan sonra READY göndermeliyiz.
    def control_server_receive(self):
        try:
            self.control_socket.recv(1024).decode("utf-8")
            print("Server received IS_READY and waiting for ready to receive...")
            self.receive_event.wait()
            self.control_socket.sendall("READY".encode("utf-8"))
            print("Server said 'i am ready to send.'")  
        except socket.error as e:
            print(f"Control Socket Error and disconnect: {e}")
            self.disconnect(keep_server=False)

    def disconnect(self, keep_server=True):
        self.client_socket.close()
        print("Client connection closed.")
        self.control_socket.close()
        print("Control connection closed.")
        if not keep_server:
            self.server_socket.close()
            print("Server connection closed.")
        else:
            self.server_socket.close()
            print("Server connection closed. Server is still running.")
            self.create_server()
            self.accept_client()







    
        

