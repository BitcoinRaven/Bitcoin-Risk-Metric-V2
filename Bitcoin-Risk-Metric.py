from datetime import date
import pandas as pd
import numpy as np
import quandl as quandl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import plotly.express as px

### Import Data
df = quandl.get("BCHAIN/MKPRU", api_key="FYzyusVT61Y4w65nFESX").reset_index()
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by="Date", inplace=True)
df = df[df["Value"] > 0]

### Last price
btcdata = yf.download(tickers='BTC-USD', period="1d", interval="1m")["Close"]
df.loc[df.index[-1]+1] = [date.today(), btcdata.iloc[-1]]
df["Date"] = pd.to_datetime(df["Date"])

### Risk
df['365'] = df['Value'].rolling(374).mean().dropna()
df["avg"] = (np.log(df.Value) - np.log(df["365"])) * df.index**0.395
df["avg"] = (df["avg"] - df["avg"].cummin()) / (df["avg"].cummax() - df["avg"].cummin())
df = df[df.index > 1500]

########## Plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

xaxis = df["Date"]
fig.add_trace(go.Scatter(x=xaxis, y=df["Value"], name="Price", line=dict(color="gold")))
fig.add_trace(go.Scatter(x=xaxis, y=df["avg"], name="Risk", line=dict(color="white")), secondary_y=True)

opacity = 0.2
for i in range(5, 0, -1):
    opacity += 0.05
    fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor="green", opacity=opacity, secondary_y=True)

opacity = 0.2
for i in range(6, 10):
    opacity += 0.1
    fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor="red", opacity=opacity, secondary_y=True)

AnnotationText = f"****Last BTC price: {round(df['Value'].iloc[-1])}**** Risk: {round(df['avg'].iloc[-1],2)}****"

fig.update_layout(template="plotly_dark")
fig.update_xaxes(title="Date")
fig.update_yaxes(title="Price", type='log', showgrid=False)
fig.update_yaxes(title="Risk", type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1)
fig.update_layout(template="plotly_dark", title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()

fig = px.scatter(df, x="Date", y="Value", color="avg", color_continuous_scale="jet")
fig.update_yaxes(title="Price", type='log', showgrid=False)
fig.update_layout(template="plotly_dark", title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()
