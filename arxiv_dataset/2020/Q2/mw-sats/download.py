#!/usr/bin/env python
"""
Download data products from GitHub.
"""
import subprocess
import yaml

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config',nargs='?',default='config.yaml',
                    help='configuration file')
args = parser.parse_args()

print("Downloading data for mw-sats...\n")

config = yaml.safe_load(open(args.config,'r'))

baseurl = config['download']['url']
release = config['download']['release']
filename = config['download']['filename']
url = '%(url)s/releases/download/%(release)s/%(filename)s'%config['download']

wget = 'wget %s'%url
tar  = 'tar -xzf %s'%filename

warning = """
WARNING: Automated download will fail until the repo is public.
Please use your browser to download files from the release page:
  %(url)s/releases/tag/%(release)s
Or directly from this url:
  %(url)s/releases/download/%(release)s/%(filename)s

Move the download file to this directory and unpack with:
  %(tar)s
"""%dict(tar=tar,**config['download'])

try:
    print(wget)
    subprocess.check_output(wget,shell=True)

    print(tar)
    subprocess.check_output(tar,shell=True)
except:
    print(warning)

