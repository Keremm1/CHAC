import socket
import json
import threading

class Client:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.receive_size = 1024 * 1024 * 10
        self.control_proccess = threading.Thread(target=self.control_client_receive)
        self.data_ready_event = threading.Event()
        self.connect_to_server()


    def connect_to_server(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.client_socket.connect((self.server_ip, self.server_port))
        print("Connected to client socket at", self.server_ip, "on port", self.server_port, "...")

        self.control_socket.connect((self.server_ip, self.server_port))
        print("Connected to control socket at", self.server_ip, "on port", self.server_port, "...")

        self.control_proccess.start()

    def receive_from_server(self):
        self.data_ready_event.set()
        self.control_proccess.join()
        self.control_proccess = threading.Thread(target=self.control_client_receive)
        print("Client waiting for data(receiving)...")
        received_data = self.client_socket.recv(self.receive_size).decode('utf-8')
        try:
            received_data = json.loads(received_data)
        except Exception as e:
            print(f"JSON Error: {e}")
        print("Client received data...", received_data)
        self.data_ready_event.clear()
        return received_data

    def send_to_server(self, data):
        self.control_server_receive()
        print("Client sending data...", data)
        try:
            data = json.dumps(data).encode('utf-8')
        except Exception as e:
            print(f"JSON Error: {e}")
        self.client_socket.sendall(data)    
        self.control_proccess.start()
        print("Client successfully sent data.")

    def control_server_receive(self):
        self.control_socket.sendall("IS_READY".encode("utf-8"))
        print("Client sended IS_READY and Client waiting for Server ready...")
        self.control_socket.recv(1024).decode("utf-8")
        print("Server said 'i am ready to receive.'")  
    
    #IS_READY aldıktan sonra READY göndermeliyiz.
    def control_client_receive(self):
        self.control_socket.recv(1024).decode("utf-8")
        print("Client received IS_READY and waiting for ready to receive...")
        self.data_ready_event.wait()
        self.control_socket.sendall("READY".encode("utf-8"))
        print("Client said 'i am ready to receive.'")  

    def disconnect(self):
        self.client_socket.close()
        print("Client connection closed.")
        self.control_socket.close()
        print("Control connection closed.")

import random
if __name__ == "__main__":
    client = Client("127.0.0.1", 5060)
    while True:
        received_data = client.receive_from_server()
        data = random.sample(range(1, 100), 10)
        if received_data == [[-1,-1,-1]]:
            break
        elif received_data == [[0,0,0]]:
            client.send_to_server("start")
            continue
        client.send_to_server(data)
    client.disconnect()
    


