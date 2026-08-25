from polars import DataFrame


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

