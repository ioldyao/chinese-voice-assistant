#!/bin/bash
# 测试天气 API 连接

API_KEY="${WEATHER_API_KEY:-test-key}"
CITY="${1:-北京}"

echo "测试天气 API..."
echo "城市: $CITY"
echo ""

curl -X GET "https://api.weather.example.com/v1/current?q=${CITY}&units=metric" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  | jq '.'
