import sqlite3

class SQLiteStorage:
    def __init__(self, db_path):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        print("Creating table", self.db_path)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY,
                params TEXT,
                value REAL
            )
        ''')
        conn.commit()
        conn.close()

    def create_trial(self, params, value):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO trials (params, value) VALUES (?, ?)', (str(params), value))
        conn.commit()
        conn.close()

    def get_all_trials(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT params, value FROM trials')
        trials = [{'params': eval(row[0]), 'value': row[1]} for row in c.fetchall()]
        conn.close()
        return trials