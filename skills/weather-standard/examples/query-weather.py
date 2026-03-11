"""
示例：查询天气信息

使用天气 API 查询指定城市的天气状况。
"""
import requests

API_KEY = "your-api-key"
BASE_URL = "https://api.weather.example.com/v1"

def query_weather(city: str) -> dict:
    """查询城市天气"""
    url = f"{BASE_URL}/current"
    params = {"q": city, "units": "metric"}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    return response.json()

if __name__ == "__main__":
    weather = query_weather("北京")
    print(f"北京天气：{weather['condition']}")
    print(f"温度：{weather['temperature']}°C")
    print(f"湿度：{weather['humidity']}%")
