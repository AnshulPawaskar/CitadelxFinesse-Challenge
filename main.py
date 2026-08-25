from py5paisa import FivePaisaClient
from dotenv import load_dotenv
from os import getenv
from asyncio import run
from Data_Extraction.data_extract import * 

load_dotenv('.env')

APP_NAME = getenv("APP_NAME")
APP_SOURCE = getenv("APP_SOURCE")
USER_ID = getenv("USER_ID")
PASSWORD = getenv("PASSWORD")
USER_KEY = getenv("USER_KEY")
ENCRYPTION_KEY = getenv("ENCRYPTION_KEY")

cred={
    "APP_NAME": APP_NAME,
    "APP_SOURCE": APP_SOURCE,
    "USER_ID": USER_ID,
    "PASSWORD": PASSWORD,
    "USER_KEY": USER_KEY,
    "ENCRYPTION_KEY": ENCRYPTION_KEY
    }

start_date = "2021-01-01"
end_date = "2025-12-31"
timeframe = "1d"

async def main():
    try:
        client = FivePaisaClient(cred=cred)
        #Store Script Book from 5Paisa
        df = await store_scripts(client, store=True, filetype="csv")
        print(df)
    except Exception as e:
        print(f"Error in the main function: {e}")

if __name__ == "__main__":
    run(main())
# req_list_ = [
#     {"Exch": "N", "ExchType": "C", "ScripData": "ITC_EQ"},
#     {"Exch": "N", "ExchType": "C", "ScripCode": "2885"}
# ]
# # print(client.fetch_market_feed_scrip(req_list_))