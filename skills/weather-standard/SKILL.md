---
name: weather-standard
display_name: 天气查询（标准版）
description: This skill should be used when the user asks to "查询天气", "今天天气怎么样", "明天会下雨吗", "北京天气如何", "天气温度", or mentions weather-related queries like temperature, humidity, wind, or forecast. Provides weather information for global locations using API integration.
version: 1.0.0
author: AI Assistant
tags: [weather, api, utility]
category: utility
requires_tools: []
---

# 天气查询技能

## Overview

Provide accurate weather information for global locations including temperature, humidity, wind speed, and weather conditions.

## Quick Start

To query weather information:
1. Identify the location (city name or coordinates)
2. Call the weather API with location parameters
3. Format and return the weather data

## Supported Queries

- Current weather: "今天北京天气怎么样"
- Forecast: "明天上海会下雨吗"
- Temperature: "现在深圳多少度"
- Humidity/Wind: "杭州湿度多大"

## API Integration

Use weather API endpoints to fetch:
- Current conditions
- Temperature (Celsius/Fahrenheit)
- Humidity percentage
- Wind speed and direction
- Weather description

## Error Handling

Handle common errors:
- Invalid location name
- API connection failure
- Rate limiting
- Missing data fields

## Additional Resources

### Reference Files

For detailed implementation guidance:
- **`references/api-spec.md`** - Complete API documentation
- **`references/error-handling.md`** - Error scenarios and solutions

### Example Files

Working examples in `examples/`:
- **`examples/query-weather.py`** - Python example
- **`examples/weather-response.json`** - Sample API response

### Scripts

Utility scripts in `scripts/`:
- **`scripts/validate-location.sh`** - Location validation
- **`scripts/test-weather-api.sh`** - API testing tool
