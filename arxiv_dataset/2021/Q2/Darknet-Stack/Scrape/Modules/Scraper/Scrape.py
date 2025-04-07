# Filename:     Scrape.py
# Description:  Logic for Crawler
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Forked from: Abhishek Singh, absingh31
# Modifed by: Edward Crowder & Jay Lansiquot

#Request libraries
import urllib3.request
from bs4 import BeautifulSoup
import sys,re,os
from http.cookiejar import CookieJar
import requests
from datetime import datetime

#Tor Connection
#Stem libraries
from stem import Signal
from stem.control import Controller
import socks, socket

#Initiating Connection
with Controller.from_port(port=9051) as controller:
    controller.authenticate("16:971AD0F466EB3C4E6017DB428E79E102E43DE0D359EFDE0B89BA7513FD")
    controller.signal(Signal.NEWNYM)

#TOR Global Variables
SOCKS_PORT = 9050  # TOR proxy port that is default from torrc, change to whatever torrc is configured to
socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", SOCKS_PORT)
socket.socket = socks.socksocket

#Perform DNS resolution through the socket
def getaddrinfo(*args):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

socket.getaddrinfo = getaddrinfo


#Scraping Utility
def Scrape(url):
    timeout = 10
    socket.setdefaulttimeout(timeout)

    #Icarus MarketPlace
    #Updated on June 29th
    #You will always need to get a valid token for via developer tools in the browser, navigate to applications and copy the cookie
    cookies = {'Icarus' : 'etvnifbcte1r275up956ulfi84'}
    req = requests.get(url,cookies=cookies)
    page = req.text
    page = BeautifulSoup(page,'html.parser')

    
    #Elite MarketPlace
    #Updated on June 29th
    #headers = {'User-Agent': 'Hi There!' }
    #req = urllib.request.Request(url,None,headers)
    #response = urllib.request.urlopen(req)
    #pageHTML = BeautifulSoup(response.read(),'html.parser')
    #page = pageHTML.find('table', {'class':'offer-vertical'})

    #Saving output
    token = re.sub(r'[^\w]', '', url)
    name = os.path.abspath('../Data/Input/icarus' + token  + '.html')
    file = open(name,'a+')
    file.write(str(page))
    file.write("<span class='META_DATA'>" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "<span>")
    file.close()

#Input
if __name__=='__main__':
    if (len(sys.argv)==2):
        url=sys.argv[1]
        Scrape(url)
    else:
        print("Invalid input")
