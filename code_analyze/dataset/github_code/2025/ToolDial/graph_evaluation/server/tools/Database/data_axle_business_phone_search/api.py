import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def business_phone_search(limit: str, query: str, packages: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find relevant businesses in the Data Axle database"
    
    """
    url = f"https://data-axle-business-phone-search.p.rapidapi.com/v1/places/search/"
    querystring = {'limit': limit, 'query': query, 'packages': packages, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "data-axle-business-phone-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

