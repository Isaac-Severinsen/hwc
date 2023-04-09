import requests
import matplotlib.pyplot as plt
import pandas as pd
import json

def get_new_access_token():
    # For this you will need a developer account with WITS: https://developer.electricityinfo.co.nz/WITS/guides
    # replace the client_id and client_secret below with your codes
    oauth_server_domain = 'api.electricityinfo.co.nz/login/'

    client_id = ''
    client_secret = ''
    token_endpoint = f'https://{oauth_server_domain}/oauth2/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret}

    response = requests.post(token_endpoint, headers=headers, data=data)

    if response.ok:
        access_token = response.json()['access_token']
    return access_token

def WITS_API_call(access_token, back = '1', forward = '1', nodes = 'ROS1101', market_type = 'E', schedules='RTD'):
    nodes_str = '%2C'.join(nodes)
    api_url = 'https://api.electricityinfo.co.nz/api/market-prices/v1/prices'

    headers = {'accept': 'application/json','Authorization': f'Bearer {access_token}'}
    if back == '1' and forward == '1':
        api_endpoint = f'{api_url}?offset=0&nodes={nodes_str}%2C&marketType={market_type}&schedules={schedules}'
    else:
        api_endpoint = f'{api_url}?offset=0&forward={forward}&back={back}&nodes={nodes_str}%2C&marketType={market_type}&schedules={schedules}'
    
    try:
        response = requests.get(api_endpoint, headers=headers)
        response.raise_for_status() # raises HTTPError if response status code is not 2xx
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401: # Access error
            # Request a new access token here, then retry the API call
            new_access_token = get_new_access_token()
            headers['Authorization'] = f'Bearer {new_access_token}'
            response = requests.get(api_endpoint, headers=headers)
        else:
            raise # Re-raise the original HTTPError if it's not an access error

    return pd.json_normalize(json.loads(response.text)[0]['prices'])

nodes_list = ['OTA2201','ROS1101','HEP0331']
# These are Auckland nodes - find yours at: 
# https://www.transpower.co.nz/our-work/industry/our-grid/maps-and-gis-data
# https://www.emi.ea.govt.nz/Wholesale/Reports/ALYRQB

access_token = get_new_access_token()
RTD = WITS_API_call(access_token, back = '48', nodes = nodes_list, schedules='RTD')      # 'RTD' - Real-time data - energy
NRSS = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='NRSS')    # 'NRSS' - Non-responseive schedule short - energy
NRSL = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='NRSL')    # 'NRSL' - Non-responseive schedule long - energy
PRSS = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='PRSS')    # 'PRSS' - Price-responseive schedule short - energy
PRSL = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='PRSL')    # 'PRSL' - Price-responseive schedule long - energy

RTD_live = WITS_API_call(access_token, nodes = ['ROS1101'], market_type = 'E', schedules='RTD')

