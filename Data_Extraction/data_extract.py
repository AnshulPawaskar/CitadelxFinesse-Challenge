from genericpath import exists
from os import makedirs
from time import sleep
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

        dup_symbols = df.filter(df["Symbol"].is_duplicated())["Symbol"].unique().to_list()
        df = df.unique(subset=["Symbol"], keep="first")
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

        # Normalize join keys so case/whitespace differences don't cause false mismatches
        stocks = stocks.with_columns(col("Symbol").str.strip_chars().str.to_uppercase().alias("_join_key"))
        scripts = scripts.with_columns(col("Name").str.strip_chars().str.to_uppercase().alias("_join_key"))
        df = stocks.join(scripts, on="_join_key", how="left").drop("_join_key")

        # Flag symbols with no match (bad join) or multiple matches (ambiguous mapping)
        missing = df.filter(col("ScripCode").is_null())
        if missing.shape[0] > 0:
            print(f"Warning: {missing.shape[0]} symbols had no matching ScripCode: {missing['Symbol'].to_list()}")

        dup_counts = df.group_by("Symbol").len().filter(col("len") > 1)
        if dup_counts.shape[0] > 0:
            print(f"Warning: {dup_counts.shape[0]} symbols matched multiple ScripCodes (ambiguous, first is kept): {dup_counts['Symbol'].to_list()}")

        df = df.unique(subset=["Symbol"], keep="first")
        return df
    except Exception as e:
        print(f"Error fetching stock meta data: {e}")
        return None

async def get_stock_data(client, stocks_df, year, half, timeframe):
    try:
        if half > 2 or half < 1:
            raise ValueError("Half must be either 1 or 2.")
        start_date = f"{year}-{'01' if half == 1 else '07'}-01"
        end_date = f"{year}-{'06' if half == 1 else '12'}-{'30' if half == 1 else '31'}"
        out_dir = f"./Data/{timeframe}/{year}/{half}"
        makedirs(out_dir, exist_ok=True)
        for row in stocks_df.iter_rows(named=True):
            symbol = row["Symbol"]
            exch = row["Exch"]
            exchtype = row["ExchType"]
            scripcode = row["ScripCode"]
            scrip_data = row["ScripData"]
            print(f"Fetching data for symbol: {symbol}, scripcode: {scripcode}, start_date: {start_date}, end_date: {end_date}")
            if scrip_data and symbol and exch and exchtype and scripcode:
                data = DataFrame(client.historical_data(exch, exchtype, scripcode, timeframe, start_date, end_date))
                print(data)
                data.write_parquet(f"{out_dir}/{scrip_data}.parquet")
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

async def data_verifier(stocks_df, year, half, timeframe):
    try:
        if half > 2 or half < 1:
            raise ValueError("Half must be either 1 or 2.")
        start_date = f"{year}-{'01' if half == 1 else '07'}-01"
        end_date = f"{year}-{'06' if half == 1 else '12'}-31"
        out_dir = f"./Data/{timeframe}/{year}/{half}"
        for row in stocks_df.iter_rows(named=True):
            symbol = row["Symbol"]
            exch = row["Exch"]
            exchtype = row["ExchType"]
            scripcode = row["ScripCode"]
            scrip_data = row["ScripData"]
            if scrip_data and symbol and exch and exchtype and scripcode:
                file_path = f"{out_dir}/{scrip_data}.parquet"
                if not exists(file_path):
                    print(f"Missing data for {scrip_data} in {file_path}")
                else:
                    total_rows = read_parquet(file_path).shape[0]
                    if total_rows == 0:
                        print(f"Data for {scrip_data} is empty in {file_path}")
    except Exception as e:
        print(f"Error verifying stock data: {e}")
        return None

async def get_ohlc(symbol, start_date, end_date, timeframe):
    try:
        year = int(start_date[:4])
        total_years = int(end_date[:4]) - year + 1
        final_data = None
        for i in range(total_years):
            current_year = year + i
            half = 1 if int(start_date[5:7]) <= 6 else 2
            for j in range(half, 3):
                file_path = f"./Data/{timeframe}/{current_year}/{j}/{symbol}.parquet"
                if exists(file_path):
                    data = read_parquet(file_path)
                    final_data = data
        return final_data
    except Exception as e:
        print(f"Error fetching OHLC data for {symbol}: {e}")
        return None

