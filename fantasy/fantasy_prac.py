import numpy

import pandas as pd
import re
import random
import requests
# from bs4 import BeautifulSoup
from selenium import webdriver
import datetime
import json

PLAYERS_INFO_FILENAME = "allPlayersInfo.json"
PLAYERS_INFO_URL = 'https://fantasy.premierleague.com/drf/bootstrap-static'
LEAGUE_URL = 'https://fantasy.premierleague.com/drf/leagues-classic-standings/'
NOIBEDDO_FPL_LEAGUE_CODE = 40735
BUET_FPL_LEAGUE_CODE = 3436



def set_global_json_to_file(url_fantasy):
    r = requests.get(url_fantasy)
    jsonResponse = r.json()
    with open(PLAYERS_INFO_FILENAME, 'w') as outfile:
        json.dump(jsonResponse, outfile)


def get_global_json_from_file():
    with open(PLAYERS_INFO_FILENAME) as json_data:
        d = json.load(json_data)
        return d


def get_league_details(league_id, page):
    league_url = LEAGUE_URL + str(league_id) + "?phase=1&le-page=1&ls-page=" + str(page)
    r = requests.get(league_url)
    json_response= r.json()

    # return json_response["standings"]["results"]

    standings = json_response["standings"]["results"]
    if not standings:
        print("no more standings found!")
        return None

    entries = []
    #
    for player in standings:
        entries.append(player["player_name"]+" "+str(player["entry"]))

    return entries

def extract_league_member_ids(league_id, pages):
    leauge_members = []
    for i in range(pages):
        leauge_members.append(get_league_details(league_id, i))
    flat_list = [item for sublist in leauge_members for item in sublist]

    return flat_list

set_global_json_to_file(PLAYERS_INFO_URL)
global_data = get_global_json_from_file()

# buet_fpl_members=extract_league_member_ids(BUET_FPL_LEAGUE_CODE, 8)
# print(len(buet_fpl_members))
# print(buet_fpl_members)

for i in range(len(global_data["teams"])):
    print(global_data["teams"][i]["name"], global_data["teams"][i]["strength"])