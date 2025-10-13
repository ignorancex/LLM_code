import time
import zmq

# Initialize ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

def main():
    while True:
        start_time = time.time()
        time.sleep(3)  # Delay for 3 seconds

        # Prepare the data to be sent
        data = {
            'bool': True,
            'int': 123,
            'str': 'Hello there!',
            # Uncomment the following line to send an image frame
            # 'image': cv2.imencode('.jpg', frame)[1].ravel().tolist()
        }
        
        # Wait for a request from the client
        socket.recv()
        
        # Send the prepared data back to the client
        socket.send_json(data)

        # Optional: Uncomment to process an image frame
        # end_time = time.time()
        # print(f"Processing time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
