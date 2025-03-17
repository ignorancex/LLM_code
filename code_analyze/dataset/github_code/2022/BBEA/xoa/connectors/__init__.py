import base64
import sys
if sys.version_info[0] < 3:
    from xoa.rest_client.restful_lib import Connection # only supported in python 2.*
else:
    from xoa.rest_client.request_lib import Connection

from xoa.commons.logger import * 


class RemoteConnectorPrototype(object):
    def __init__(self, target_url, credential, **kwargs):
        self.url = target_url
        self.credential = credential
        
        if "timeout" in kwargs:
            self.timeout = kwargs['timeout']
        else:
            self.timeout = 100000

        if "num_retry" in kwargs:
            self.num_retry = kwargs['num_retry']
        else:
            self.num_retry = 100

        self.conn = Connection(target_url, timeout=self.timeout)
        
        self.headers = {'Content-Type':'application/json', 'Accept':'application/json'}
        auth_key = base64.b64encode(self.credential.encode('utf-8'))
        auth_key = "Basic {}".format(auth_key.decode("utf-8"))
        self.headers['Authorization'] = auth_key