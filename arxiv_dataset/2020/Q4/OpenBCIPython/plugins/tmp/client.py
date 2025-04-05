import json
import cPickle as pickle
import socket
import sys

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('localhost', 5678)
message = [45,67,89]

try:

    data_string = json.dumps(message)

    # Send data
    print >>sys.stderr, 'sending "%s"' % data_string
    sent = sock.sendto(data_string, server_address)

    # # Receive response
    # print >>sys.stderr, 'waiting to receive'
    # data, server = sock.recv(4096)
    # print >>sys.stderr, 'received "%s"' % data

finally:
    print >>sys.stderr, 'closing socket'
    sock.close()