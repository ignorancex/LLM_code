
import sqlite3



class BirdDatabaseSearcher:
    def __init__(self):
        pass

    def search(self, sql_query: str, db_path: str):
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
