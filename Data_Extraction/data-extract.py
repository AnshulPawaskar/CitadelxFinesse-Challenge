from py5paisa import FivePaisaClient
from dotenv import load_dotenv
from os import getenv
from polars import DataFrame
from asyncio import run

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

async def store_scripts(client, store=True, filetype="csv"):
    try:
        data = client.get_scrips()
        df = DataFrame(data)
        if store:
            if filetype == "csv":
                df.write_csv("script_details.csv")
            elif filetype == "parquet":
                df.write_parquet("script_details.parquet")
        return df
    except Exception as e:
        print(f"Error storing scripts: {e}")
        return None

async def get_scripts(filetype="csv", filename="script_details"):
    try:
        data = DataFrame.read_csv(f"{filename}.{filetype}")
        return data
    except Exception as e:
        print(f"Error fetching scripts: {e}")
        return None

async def get_index(indices=["nse_100", "nse_midcap_100", "nse_smallcap_100"]):
    try:
        df = None
        for i in indices:
            if df is None:
                df = DataFrame.read_csv(f"{i}.csv")
            else:
                df = df + DataFrame.read_csv(f"{i}.csv")
        return df
    except Exception as e:
        print(f"Error fetching index data: {e}")
        return None

async def main():
    try:
        client = FivePaisaClient(cred=cred)
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