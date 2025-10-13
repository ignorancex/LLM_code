import os
import argparse

import numpy as np

import pyspark
from pyspark.sql.functions import trim
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

from rouge_score import rouge_scorer

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="")
    parser.add_argument("--dataset", type=str, help="")
    parser.add_argument("--driver_memory", type=str, default="16g", help="")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown = read_args()

    conf = pyspark.SparkConf()
    conf.set('spark.driver.memory', args.driver_memory)
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    task_output_dir = os.path.join(args.data_root)

    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)

    max_length = 20000
    data_prefixes = ['train', 'val', 'test']

    if args.dataset == 'bigpatent':
        for prefix in data_prefixes:
            df = spark.read.json(os.path.join(args.data_root, 'source', prefix, '*', '*.gz'))#.repartition(args.partitions, "article_id")

            df = df \
                .withColumn("document_len", F.size(F.split(F.col("description"), " "))) \
                .withColumn("summary_len", F.size(F.split(F.col("abstract"), " "))) \
                .where(F.col('document_len') > max_length) \
                .where(F.col('summary_len') > 50) \
                .select(
                    df.publication_number.alias("article_id"),
                    df.description.alias("document"),
                    df.abstract.alias("summary"),
                    "document_len",
                    "summary_len")
            
            print(f'Total rows: {df.count()}')
            print(f'Columns: {df.columns}')

            df.write.json(
                path=os.path.join(task_output_dir, prefix),
                mode="overwrite")

            print(f"Finished writing {prefix} split to {task_output_dir}")
    else:
        train_data = os.path.join(args.data_root, 'train.txt')
        val_data = os.path.join(args.data_root, 'val.txt')
        test_data = os.path.join(args.data_root, 'test.txt')

        data_paths = [train_data, val_data, test_data]

        for data_path, prefix in zip(data_paths, data_prefixes):
            df = spark.read.json(data_path)#.repartition(args.partitions, "article_id")
            #df = df.repartition(args.partitions, "article_id")

            df = df \
                .withColumn('document', trim(F.array_join(F.col('article_text'), " "))) \
                .withColumn('summary', trim(F.array_join(F.col('abstract_text'), " "))) \
                .withColumn("summary", F.regexp_replace("summary", "<\/?S>", "")) \
                .withColumn("document_len", F.size(F.split(F.col("document"), " "))) \
                .withColumn("summary_len", F.size(F.split(F.col("summary"), " "))) \
                .where(F.col('document_len') > max_length) \
                .where(F.col('summary_len') > 50) \
                .select(
                    "article_id",
                    "document",
                    "summary",
                    "document_len",
                    "summary_len")
            
            print(f'Total rows: {df.count()}')
            print(f'Columns: {df.columns}')

            df.write.json(
                path=os.path.join(task_output_dir, prefix),
                mode="overwrite")

            print(f"Finished writing {prefix} split to {task_output_dir}")

if __name__ == "__main__":
    main()
