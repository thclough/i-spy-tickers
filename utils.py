
import requests

def get_json_dict(source):
    """retrieve json from input url and creates a dictionary
    
    Args:
        url (str) : json location

    Returns:
        json_dict (dict) : json in dictionary format
    """

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'

    response = requests.get(source, headers = {'User-Agent': user_agent})

    if response.status_code != 200:
        raise Exception("could not locate ticker_dict JSON source")
        
    json_dict = response.json()
    return json_dict