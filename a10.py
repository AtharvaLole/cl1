import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("datasets\\assignment_10\\City_Air_Quality.csv")

data["Date"]=pd.to_datetime(data["Date"])
data=data.dropna(subset=["AQI"])
data=data.sort_values("Date")

data=data[data["City"]=="Delhi"]
data=data.head(100)

plt.plot(data["Date"],data["AQI"])
plt.xlabel("date")
plt.ylabel("aqi")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.plot(data["Date"],data["CO"],label="CO")
plt.plot(data["Date"],data["PM2.5"],label="PM2.5")
plt.plot(data["Date"],data["PM10"],label="PM10")
plt.legend()
plt.xticks(rotation=45)
plt.show()

plt.bar(data["Date"].head(10),data["AQI"].head(10))
plt.title("AQI comparison")
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.boxplot(data["AQI"])
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.show()

plt.scatter(data["PM2.5"],data["AQI"])
plt.xlabel("pm2.5")
plt.ylabel("aqi")
plt.xticks(rotation=45)
plt.show()