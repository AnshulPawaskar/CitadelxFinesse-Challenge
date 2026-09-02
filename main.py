from py5paisa import FivePaisaClient
from dotenv import load_dotenv
from os import getenv
from asyncio import run
from datetime import datetime
from Data_Extraction.data_extract import * 

load_dotenv('.env')

APP_NAME = getenv("APP_NAME")
APP_SOURCE = getenv("APP_SOURCE")
USER_ID = getenv("USER_ID")
PASSWORD = getenv("PASSWORD")
USER_KEY = getenv("USER_KEY")
ENCRYPTION_KEY = getenv("ENCRYPTION_KEY")
RESPONSE_TOKEN = getenv("REQUEST_TOKEN")

cred={
    "APP_NAME": APP_NAME,
    "APP_SOURCE": APP_SOURCE,
    "USER_ID": USER_ID,
    "PASSWORD": PASSWORD,
    "USER_KEY": USER_KEY,
    "ENCRYPTION_KEY": ENCRYPTION_KEY
    }

start_date = "2020-01-01"
end_date = "2026-06-30"
timeframe = "1d"

def year_half(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.year, 1 if dt.month <= 6 else 2

async def main():
    try:
        client = FivePaisaClient(cred=cred)
        try:
            client.get_oauth_session(RESPONSE_TOKEN)
            access_token = client.get_access_token()
            client.set_access_token(access_token, USER_ID)
        except Exception as auth_err:
            print(f"Warning: authentication failed, continuing without a live session: {auth_err}")

        #Store Script Book from 5Paisa
        # df = await store_scripts(client, store=True, filetype="parquet")

        #Fetch Script Book from local storage
        scripts = await get_scripts(filetype="parquet", filename="script_details")

        #List of tradable stocks
        stocks = await get_index(indices=["nse_100", "nse_midcap_100", "nse_smallcap_100"])

        #Fetch stock meta data
        df_final = await get_stock_meta_data(scripts=scripts, stocks=stocks) 
        # print(df_final)
        # scriptcode = df_final[0]["ScripCode"][0]
        # print(scriptcode)
        # data = client.historical_data("N", "C", scriptcode, timeframe, start_date, end_date)
        # print(data)

        #Get Stock Historical Data
        start_year, start_half = year_half(start_date)
        end_year, end_half = year_half(end_date)

        year, half = start_year, start_half
        while (year, half) <= (end_year, end_half):
            await get_stock_data(client=client, stocks_df=df_final, year=year, half=half, timeframe=timeframe)
            if half == 1:
                half = 2
            else:
                year, half = year + 1, 1
        print("Data extraction and storage completed successfully.")

        #Verify Stock Historical Data
        # year, half = start_year, start_half
        # while (year, half) <= (end_year, end_half):
        #     await data_verifier(stocks_df=df_final, year=year, half=half, timeframe=timeframe)
        #     if half == 1:
        #         half = 2
        #     else:
        #         year, half = year + 1, 1
    except Exception as e:
        print(f"Error in the main function: {e}")

if __name__ == "__main__":
    run(main())
