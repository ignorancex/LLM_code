import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shakespeare(x_funtranslations_api_secret: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shakespeare Translator"
    x_funtranslations_api_secret: API Key ( Get yours here http://funtranslations.com/api/shakespeare )
        text: Text to convert to Shakespeare style English.
        
    """
    url = f"https://shakespeare.p.rapidapi.com/shakespeare.json"
    querystring = {'X-FunTranslations-Api-Secret': x_funtranslations_api_secret, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shakespeare.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

