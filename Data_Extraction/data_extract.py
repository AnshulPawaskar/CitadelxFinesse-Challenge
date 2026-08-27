from polars import DataFrame, read_csv, read_parquet, concat, col


async def store_scripts(client, store=True, filetype="csv"):
    try:
        data = client.get_scrips()
        df = DataFrame(data)
        if store:
            if filetype == "csv":
                df.write_csv("./ScriptBook/script_details.csv")
            elif filetype == "parquet":
                df.write_parquet("./ScriptBook/script_details.parquet")
        return df
    except Exception as e:
        print(f"Error storing scripts: {e}")
        return None

async def get_scripts(filetype="csv", filename="script_details"):
    try:
        if filetype == "csv":
            data = read_csv(f"./ScriptBook/{filename}.csv")
        elif filetype == "parquet":
            data = read_parquet(f"./ScriptBook/{filename}.parquet")
        return data
    except Exception as e:
        print(f"Error fetching scripts: {e}")
        return None

async def get_index(indices=["nse_100", "nse_midcap_100", "nse_smallcap_100"]):
    try:
        df = None
        for i in indices:
            if df is None:
                df = read_csv(f"./Stocks/{i}.csv")
            else:
                df = concat([df, read_csv(f"./Stocks/{i}.csv")], how="vertical")
        return df
    except Exception as e:
        print(f"Error fetching index data: {e}")
        return None

async def get_stock_meta_data(scripts, stocks):
    try:
        scripts = scripts.filter(
            (col("Exch") == "N") &
            (col("ExchType") == "C") &
            (col("Series") == "EQ")
        ).select(["Name", "Exch", "ExchType", "ScripCode", "ScripData"])
        df = stocks.join(scripts, left_on="Symbol", right_on="Name", how="left")
        return df
    except Exception as e:
        print(f"Error fetching stock meta data: {e}")
        return None

async def get_stock_data(client, stocks_df, year, half, timeframe):
    try:
        if half > 2 or half < 1:
            raise ValueError("Half must be either 1 or 2.")
        start_date = f"{year}-{'01' if half == 1 else '07'}-01"
        end_date = f"{year}-{'06' if half == 1 else '12'}-31"
        for i in stocks_df.iter_rows():
            scrip_data = i[3]  # Assuming ScripData is the 5th column
            if scrip_data:
                data = DataFrame(client.fetch_historical_data(scrip_data, start_date, end_date, timeframe))
                data.write_parquet(f"./Data/{year}/{half}/{scrip_data}.parquet")
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None



