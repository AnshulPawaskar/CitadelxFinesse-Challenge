from talib import *

async def calculate_sma(data, timeperiod):
    return SMA(data, timeperiod)

async def calculate_ema(data, timeperiod):
    return EMA(data, timeperiod)

async def calculate_rsi(data, timeperiod):
    return RSI(data, timeperiod)

async def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    return MACD(data, fastperiod, slowperiod, signalperiod)

async def calculate_bollinger_bands(data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    return BBANDS(data, timeperiod, nbdevup, nbdevdn, matype)

async def calculate_stochastic(data, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    return STOCH(data, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)

async def calculate_adx(data, timeperiod=14):
    return ADX(data, timeperiod)

async def calculate_momentum(data, timeperiod=10):
    return MOM(data, timeperiod)

async def calculate_obv(data):
    return OBV(data)

async def calculate_cci(data, timeperiod=14):
    return CCI(data, timeperiod)

async def calculate_atr(data, timeperiod=14):
    return ATR(data, timeperiod)

async def calculate_ad(data, timeperiod=14):
    return AD(data, timeperiod)

async def calculate_adosc(data, fastperiod=3, slowperiod=10):
    return ADOSC(data, fastperiod, slowperiod)

