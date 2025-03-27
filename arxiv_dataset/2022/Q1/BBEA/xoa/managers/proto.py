import base64
from xoa.commons.logger import * 
from xoa.managers.db_mgr import get_database_manager 


class ManagerPrototype(object):

    def __init__(self, mgr_type):
        self.type = mgr_type
        self.dbm = get_database_manager()

    def get_credential(self):
        database = self.dbm.get_db()        
        return database['credential']

    def get_train_jobs(self):
        database = self.dbm.get_db()
        if 'train_jobs' in database:       
            return database['train_jobs']
        else:
            return []        

    def get_hpo_jobs(self):
        database = self.dbm.get_db()
        if 'hpo_jobs' in database:       
            return database['hpo_jobs']
        else:
            return []  

    def get_users(self):
        database = self.dbm.get_db()
        if 'users' in database:        
            return database['users']
        else:
            return []

    def save_db(self, key, data):
        database = self.dbm.get_db()            
        database[key] = data
        self.dbm.save(database)

    def authorize(self, auth_key):
        key = auth_key.replace("Basic ", "")
        try:
            u_pw = base64.b64decode(key).decode('utf-8')
            #debug("User:Password = {}".format(u_pw))
            if ":" in u_pw:
                tokens = u_pw.split(":")
                #debug("Tokens: {}".format(tokens))
                for u in self.get_users():
                    if tokens[0] in u and u[tokens[0]] == tokens[1]:
                        return True
            elif u_pw == self.get_credential():
                return True
            else:
                return False

        except Exception as ex:
            debug("Auth key {} decoding error: {}".format(key, ex))
        return False
		
		