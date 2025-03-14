import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qrcode_image_copy(data: str='www.youtube.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET request to qr code image"
    
    """
    url = f"https://qrcode-api7.p.rapidapi.com/qr_code"
    querystring = {}
    if data:
        querystring['data'] = data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-api7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

