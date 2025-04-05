import Queue
import json
import socket
import sys
import threading
import init_buffer as buf

import numpy as np

from RingBuffer import RingBuffer


class UDPServer(threading.Thread):
    def __init__(self, threadID, input_buffer, port, receiver_port, ip="localhost", verbose=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.isRun = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.server_address = (self.ip, self.port)
        self.lock = threading.Lock()
        self.previous_buffer = [-1,-1,-1]
        self.buffer = input_buffer
        self.receiver_port = receiver_port
        self.verbose = verbose
        self.file_descriptor = ""
        if self.verbose:
            self.file_descriptor = ""
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.server_address)
        except:
            print ("Error: while starting up udp server")

    def run(self):
        print ('starting up on %s port %s \n' % self.server_address)
        self.call_back_handler()
        self.socket.close()
        print ("Exiting " + self.threadID)

    def get_previous_buffer(self):
        return self.previous_buffer

    def get_next_point(self):
        # self.lock.acquire()
        # if not self.buffer.empty():
        #     data = self.buffer.pop_window(10)
        #     self.lock.release()
        #     return data
        # else:
        #     self.lock.release()
        return self.previous_buffer

    # def retrieve_data(self):
    #     while self.isRun:
    #         data, address = self.socket.recvfrom(self.receiver_port)
    #         data = json.loads(data)
    #         result = []
    #         if data:
    #             self.lock.acquire()
    #             if self.verbose:
    #                 print (data)
    #             self.buffer.append(data)
    #             self.previous_buffer = data
    #             # result.append(data)
    #             # result = np.asarray(result)
    #             # np.savetxt(self.file_descriptor, result, delimiter=',', fmt='%.18e')
    #             self.lock.release()

    def call_back_handler(self):
        while self.isRun:
            # print "-----------------------------------------------"
            data, address = self.socket.recvfrom(self.receiver_port)
            data = json.loads(data)
            result = []
            if data:
                # channal_result = buf.channel_data
                try:
                    # self.lock.acquire()
                    if self.verbose:
                        print (data)
                    self.buffer.append(data)
                    self.previous_buffer = data
                    #     with open("/home/runge/openbci/git/OpenBCI_Python/build/dataset/result_bicep_new.csv", 'a') as f:
                    #         for i in self.previous_buffer:
                    #             channal_result += str(i)
                    #             channal_result += ","
                    #         channal_result[-1].replace(",", "")
                    #         channal_result += '\n'
                    #         f.write(channal_result)
                except:
                    # print (channal_result)
                    print (self.previous_buffer)
                    pass
                # result.append(data)
                # result = np.asarray(result)
                # np.savetxt(self.file_descriptor, result, delimiter=',', fmt='%.18e')
                # self.lock.release()
        # if self.verbose:
        #     with open("/home/runge/openbci/git/OpenBCI_Python/build/dataset/kincet_anagles/kinect_angles.csv", 'a')\
        #             as self.file_descriptor:
        #         self.retrieve_data()
        # else:
        # self.retrieve_data()

# kinect_angles = RingBuffer(20, dtype=list)
# ip = "0.0.0.0"
# port = 8889
# receiver_port = 4096
# thread = UDPServer("udp_server", kinect_angles, port, receiver_port, ip, False)
# thread.start()
# thread.isRun = True
# thread.join()

