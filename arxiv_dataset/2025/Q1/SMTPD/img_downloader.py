from urllib.request import urlretrieve
import urllib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import threadpool
import requests as req

img_path = 'img_yt/'


new_csv = 'basic_content.csv'
# excel_path = './data_new/data_pd.xlsx'

# if source == 'bili' or 'bili_new':
path_column = 2
name_column = 0


def downloader(i):
    file_path = img_path + str(all_lines[i][name_column]) + '.jpg'
    if os.path.exists(file_path):
        return
    else:
        try:
            url1 = all_lines[i][path_column]
            # url1 = "http://" + url1.lstrip("https://")
            # print(req.get(url1, verify=False))
            urlretrieve(url1, file_path)
            print(i*100/num, '%')
        except Exception as e:
            # print("failed %d" % i)
            # print(e)
            pass


# 添加请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Host": "i.ytimg.com",
    # "Connection": "keep-alive",
    # "Sec-Fetch-Mode": "navigate",
    # "Sec-Fetch-Site": "cross-site",
    # "Upgrade-Insecure-Requests": "1",
    # "Referer": "https://www.youtube.com",
}

# headers = [(key, val) for key, val in headers.items()]
# opener = urllib.request.build_opener()
# for header in headers:
#     opener.addheaders = [header]
#
# urllib.request.install_opener(opener)

if __name__=='__main__':

    # xlsx_to_csv_pd()

    all_lines = pd.read_csv(new_csv)
    all_lines = np.array(all_lines)
    # 打开文件位置
    os.makedirs(img_path, exist_ok=True)

    num = len(all_lines)

    # start_time = time.time()
    # 定义了一个线程池，最多创建10个线程
    pool = threadpool.ThreadPool(10)
    # 创建要开启多线程的函数，以及函数相关参数和回调函数，其中回调数可以不写，default是none
    requests = threadpool.makeRequests(downloader, range(num))
    # 将所有要运行多线程的请求扔进线程池
    [pool.putRequest(req) for req in requests]
    # 所有的线程完成工作后退出
    pool.wait()


