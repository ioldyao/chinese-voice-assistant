# Weather API Specification

## Base URL
```
https://api.weather.example.com/v1
```

## Authentication
```python
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
```

## Endpoints

### Current Weather
```
GET /current?q={city}&units=metric
```

### Forecast
```
GET /forecast?q={city}&days={days}
```

## Response Format
```json
{
  "location": "Beijing",
  "temperature": 25,
  "humidity": 60,
  "wind_speed": 12,
  "condition": "Partly Cloudy"
}
```
