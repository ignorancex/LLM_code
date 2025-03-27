from dataset import  get_all_data, load_data, data_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_http_context = ssl._create_unverified_context
get_all_data()