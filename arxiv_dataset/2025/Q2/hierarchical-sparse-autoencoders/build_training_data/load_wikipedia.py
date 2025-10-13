import pyarrow.parquet as pq
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
import concurrent.futures

db_name = "" # SQL database name, e.g. 'interp'
username = "" # DB username, e.g. 'muchane'
url = "" # DB url/IP address, e.g. 127.0.0.1

def get_parquet_files(base_dir):
    files = []
    for subdir, _, filelist in os.walk(base_dir):
        if subdir.startswith(base_dir):
            lang = subdir.split('.')[-1]
            for file in filelist:
                if file.endswith('.parquet'):
                    files.append((os.path.join(subdir, file), lang))
    return files

def load_single_file(file, lang, conn_str):
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    table = pq.read_table(file)
    df = table.to_pandas()
    df['lang'] = lang
    df.rename(columns={'id': 'article_id'}, inplace=True)
    records = df.to_dict('records')
    query = """
        INSERT INTO wikipedia_acts (article_id, url, title, text, lang)
        VALUES (%(article_id)s, %(url)s, %(title)s, %(text)s, %(lang)s)
    """
    execute_batch(cursor, query, records)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Loaded data from {file}")

def load_data_to_postgres(parquet_files, conn_str):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_file, file, lang, conn_str) for file, lang in parquet_files]
        for future in concurrent.futures.as_completed(futures):
            future.result()

conn_str = f"dbname={db_name} user={username} host={url}"
parquet_files = get_parquet_files('wikipedia')
load_data_to_postgres(parquet_files, conn_str)

print("Data loading complete!")
