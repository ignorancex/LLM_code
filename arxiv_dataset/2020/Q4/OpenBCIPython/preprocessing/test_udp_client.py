import json
import socket
import sys

# Create a UDP socket
import threading

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('localhost', 8889)
message =  "34,56,89"
message =  json.dumps([34,56,89])

try:
    while True:
    # Send data
        print >>sys.stderr, 'sending "%s"' % message
        sent = sock.sendto(message, server_address)
        threading._sleep(1)

finally:
    print >>sys.stderr, 'closing socket'
    sock.close()