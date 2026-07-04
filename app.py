
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import json
import random
import math
import os
import time
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from engine.score_engine import score_crop
from engine.recommendation_engine import recommend_top_k, compute_final_score
from engine.rotation_engine import get_rotation_score

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Flask app
app.config['DEBUG'] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Session with improved connection pooling and retry strategy
def create_session():
    """Create an HTTP session with connection pooling and retry strategy"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait time between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["HEAD", "GET", "POST"]  # HTTP methods to retry
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session instance
api_session = create_session()

# Simple in-memory cache for API responses
weather_cache = {}
soil_cache = {}
CACHE_DURATION = 300  # 5 minutes in seconds
geocode_cache = {}

def is_cache_valid(timestamp):
    return time.time() - timestamp < CACHE_DURATION

def get_cache_key(lat, lon, extra=None):
    # Round coordinates to reduce cache misses for nearby locations
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    if extra:
        return f"{lat_rounded},{lon_rounded},{extra}"
    return f"{lat_rounded},{lon_rounded}"


def geocode_location(city=None, state=None, country="India"):
    parts = [str(part).strip() for part in (city, state, country) if part]
    if not parts:
        return None

    key = tuple(parts)
    cached = geocode_cache.get(key)
    if cached:
        coords, timestamp = cached
        if is_cache_valid(timestamp):
            return coords

    query = ", ".join(parts)
    try:
        print(f"📍 Geocoding manual location: {query}")
        resp = api_session.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "farm-tycoon/1.0 (https://github.com)"},
            timeout=10
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            first = results[0]
            lat = float(first.get("lat"))
            lon = float(first.get("lon"))
            geocode_cache[key] = ((lat, lon), time.time())
            return lat, lon
    except Exception as exc:
        print(f"⚠️ Geocoding failed for '{query}': {exc}")

    return None

# Configuration - Use environment variables for API keys
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '')

# ---------------- Load crop data -----------------
DATA_DIR = os.path.join(app.root_path, "data")

with open(os.path.join(DATA_DIR, "crops.json"), "r") as f:
    crop_data = json.load(f)
@app.route("/data/<path:filename>")
def serve_data_file(filename: str):
    """Serve JSON/data assets for frontend requests."""
    return send_from_directory(DATA_DIR, filename)
# Build a lookup by crop name for quick access in explanation builder
CROP_DB = {c.get('Crop'): c for c in (crop_data or []) if isinstance(c, dict) and c.get('Crop')}

# Translation helper configuration (lazy optional import)
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "gu": "Gujarati"
}

try:
    from googletrans import Translator  # type: ignore

    translator = Translator()
    TRANSLATION_AVAILABLE = True
except Exception:
    translator = None
    TRANSLATION_AVAILABLE = False


SOIL_TYPE_CHEMISTRY = {
    "Sandy": {"ph": 6.0, "om": 1.2},
    "Sandy Loam": {"ph": 6.2, "om": 1.6},
    "Sandy Clay": {"ph": 6.4, "om": 1.9},
    "Sandy Clay Loam": {"ph": 6.6, "om": 2.4},
    "Loam": {"ph": 6.8, "om": 2.1},
    "Clay": {"ph": 7.3, "om": 3.4},
    "Clay Loam": {"ph": 7.0, "om": 2.7},
    "Silty Clay": {"ph": 7.1, "om": 3.0},
    "Silt Loam": {"ph": 6.7, "om": 2.3},
    "Alluvial": {"ph": 7.1, "om": 2.8},
    "Black": {"ph": 7.4, "om": 3.2},
    "Red": {"ph": 6.2, "om": 1.8},
    "Laterite": {"ph": 5.8, "om": 1.5},
    "Desert": {"ph": 7.6, "om": 0.9},
    "Mixed": {"ph": 6.7, "om": 2.2}
}

SOILGRIDS_WMS_BASE = "https://maps.isric.org/mapserv"


def estimate_soil_chemistry(soil_type: Optional[str], clay: Optional[float], sand: Optional[float], silt: Optional[float]) -> Dict[str, float]:
    """Fallback soil chemistry estimates when live services fail."""

    if soil_type:
        baseline = SOIL_TYPE_CHEMISTRY.get(soil_type)
        if baseline:
            return baseline

    clay = clay if clay is not None else 0.0
    sand = sand if sand is not None else 0.0
    silt = silt if silt is not None else 0.0
    total = max(clay + sand + silt, 1.0)
    clay_ratio = clay / total
    sand_ratio = sand / total

    # Clay-heavy soils trend alkaline with richer organic matter; sandy soils skew acidic/lean
    ph = 6.5 + (clay_ratio * 0.8) - (sand_ratio * 0.4)
    organic_matter = 2.0 + (clay_ratio * 1.2) - (sand_ratio * 0.7)
    return {
        "ph": round(max(4.8, min(7.8, ph)), 1),
        "om": round(max(0.5, min(4.0, organic_matter)), 1)
    }


def sample_soilgrids_rest_api(lat: float, lon: float) -> Optional[Dict[str, dict]]:
    """
    Fetch soil properties (clay, sand, silt, ph, soc) from SoilGrids REST API in single call.
    Much faster than WMS because it retrieves all properties at once.
    Returns: {"clay": {...}, "sand": {...}, "silt": {...}, "ph": {...}, "soc": {...}}
    """
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon": lon,
            "lat": lat,
            "depth": "0-5cm",
            "property": "clay,sand,silt,phh2o,soc",
            "output": "json"
        }
        response = api_session.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        if not data or "properties" not in data:
            return None
        
        props = data["properties"]
        result = {}
        
        # Process clay, sand, silt (0-5cm layer)
        for prop_name in ["clay", "sand", "silt"]:
            if prop_name in props:
                layers = props[prop_name]
                if layers and "0-5cm" in layers:
                    layer_data = layers["0-5cm"]
                    if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
                        value = layer_data[0].get("value") if isinstance(layer_data[0], dict) else layer_data[0]
                        if value is not None:
                            result[prop_name] = {"value": value, "source": "REST"}
        
        # Process pH (0-5cm layer, stored as phh2o)
        if "phh2o" in props:
            layers = props["phh2o"]
            if layers and "0-5cm" in layers:
                layer_data = layers["0-5cm"]
                if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
                    value = layer_data[0].get("value") if isinstance(layer_data[0], dict) else layer_data[0]
                    if value is not None:
                        result["ph"] = {"value": value, "source": "REST"}
        
        # Process SOC (0-5cm layer)
        if "soc" in props:
            layers = props["soc"]
            if layers and "0-5cm" in layers:
                layer_data = layers["0-5cm"]
                if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
                    value = layer_data[0].get("value") if isinstance(layer_data[0], dict) else layer_data[0]
                    if value is not None:
                        result["soc"] = {"value": value, "source": "REST"}
        
        if result:
            print(f"[soilgrids-rest] Retrieved {len(result)} properties for {lat:.4f},{lon:.4f}")
            return result
        
        return None
    except Exception as exc:
        print(f"[soilgrids-rest] REST API failed: {exc}")
        return None


def sample_soilgrids_wms(map_path: str, layer_name: str, lat: float, lon: float) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Fetch a SoilGrids pixel via WMS GetFeatureInfo with small-radius search (fallback after REST)."""

    # Ensure coordinates are within valid bounds
    def clamp_lat(val: float) -> float:
        return max(-89.9999, min(89.9999, val))

    search_radii = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    directions = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]

    for radius in search_radii:
        delta = max(0.02, radius + 0.02)
        for dx, dy in directions:
            if radius == 0.0 and (dx, dy) != (0.0, 0.0):
                continue

            sample_lat = clamp_lat(lat + (dy * radius))
            sample_lon = max(-180.0, min(180.0, lon + (dx * radius)))
            minx = max(-180.0, sample_lon - delta)
            maxx = min(180.0, sample_lon + delta)
            miny = clamp_lat(sample_lat - delta)
            maxy = clamp_lat(sample_lat + delta)

            width = height = 256
            try:
                x = int(round(((sample_lon - minx) / (maxx - minx)) * (width - 1))) if maxx != minx else width // 2
                y = int(round(((maxy - sample_lat) / (maxy - miny)) * (height - 1))) if maxy != miny else height // 2
            except ZeroDivisionError:
                x = width // 2
                y = height // 2

            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            params = {
                'map': map_path,
                'SERVICE': 'WMS',
                'VERSION': '1.1.1',
                'REQUEST': 'GetFeatureInfo',
                'LAYERS': layer_name,
                'QUERY_LAYERS': layer_name,
                'INFO_FORMAT': 'application/geo+json',
                'SRS': 'EPSG:4326',
                'X': x,
                'Y': y,
                'WIDTH': width,
                'HEIGHT': height,
                'BBOX': f"{minx},{miny},{maxx},{maxy}",
                'FORMAT': 'image/png',
                'STYLES': '',
                'FEATURE_COUNT': 1
            }

            try:
                response = api_session.get(SOILGRIDS_WMS_BASE, params=params, timeout=25)
                if response.status_code != 200:
                    continue
                content = response.text.strip()
                if not content or not content.startswith('{'):
                    continue
                payload = response.json()
                features = payload.get('features')
                if features:
                    props = features[0].get('properties') or {}
                    raw_val = props.get('pixel_value')
                    if raw_val is not None:
                        sampled = (round(sample_lat, 5), round(sample_lon, 5))
                        return float(raw_val), sampled
            except Exception as exc:
                print(f"[soilgrids] WMS lookup failed ({layer_name}): {exc}")
                continue

    return None, None


# ---------------- Multiple Weather APIs -----------------
def get_openmeteo_weather(lat, lon, date=None):
    """Enhanced Open-Meteo API with better error handling and multiple endpoints"""
    endpoints = [
        "https://api.open-meteo.com/v1/forecast",
        "https://archive-api.open-meteo.com/v1/archive"
    ]
    
    try:
        # Use current forecast for recent dates
        if not date or date >= datetime.today().strftime("%Y-%m-%d"):
            url = endpoints[0]
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,precipitation',
                'daily': 'temperature_2m_max,precipitation_sum',
                'timezone': 'auto',
                'forecast_days': 1
            }
        else:
            url = endpoints[1]
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': date,
                'end_date': date,
                'daily': 'temperature_2m_max,precipitation_sum',
                'timezone': 'auto'
            }
        
        print("🌤️ Trying Open-Meteo API")

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"❌ Open-Meteo API response error: {response.status_code}")
            return None

        data = response.json()

        temp_value = None
        rainfall_value = None

        daily = data.get('daily', {}) if isinstance(data, dict) else {}
        if daily:
            temps = daily.get('temperature_2m_max') or daily.get('temperature_2m_mean') or []
            rain = daily.get('precipitation_sum') or daily.get('rain_sum') or []
            if temps:
                temp_value = temps[0]
            if rain:
                rainfall_value = rain[0]

        if temp_value is None:
            current = data.get('current') or data.get('current_weather') or {}
            temp_value = current.get('temperature_2m') or current.get('temperature')

        if rainfall_value is None:
            hourly = data.get('hourly', {})
            if hourly:
                precip = hourly.get('precipitation') or hourly.get('rain')
                if precip:
                    rainfall_value = precip[0]

        temp_value = float(temp_value) if temp_value is not None else 25.0
        rainfall_value = float(rainfall_value) if rainfall_value is not None else 0.0

        print(f"✅ Open-Meteo: {temp_value}°C, {rainfall_value}mm (daily)")
        return {
            "temp": round(temp_value, 1),
            "rainfall": round(rainfall_value, 1),
            "source": "Open-Meteo"
        }
    except Exception as e:
        print(f"❌ Open-Meteo failed: {e}")
        return None

def combine_weather_data(weather_sources):
    if not weather_sources:
        raise Exception("No weather data available")
    temps = [w.get('temp', 25) for w in weather_sources if w and isinstance(w, dict) and 'temp' in w and w['temp'] is not None]
    rainfalls = [w.get('rainfall', 0) for w in weather_sources if w and isinstance(w, dict) and 'rainfall' in w and w['rainfall'] is not None]
    avg_temp = sum(temps) / len(temps) if temps else 25
    avg_rainfall = sum(rainfalls) / len(rainfalls) if rainfalls else 0
    temp_anomalies = [abs(t - avg_temp) / max(avg_temp, 1) > 0.2 for t in temps if t is not None]
    rain_anomalies = [abs(r - avg_rainfall) / (avg_rainfall + 1) > 0.3 for r in rainfalls if r is not None]
    sources = [w.get('source', 'Unknown') for w in weather_sources if w and isinstance(w, dict)]
    print(f"🌡️ Combined weather: {avg_temp:.1f}°C, {avg_rainfall:.1f}mm from {', '.join(sources)}")
    return {
        "temp": round(avg_temp, 1),
        "rainfall": round(avg_rainfall, 1),
        "sources": sources,
        "temp_anomalies": any(temp_anomalies),
        "rain_anomalies": any(rain_anomalies)
    }

def get_weather(lat, lon, date=None):
    """Enhanced weather fetching with concurrent API calls, smart fallbacks, and caching"""
    # Check cache first
    cache_key = get_cache_key(lat, lon, date)
    if cache_key in weather_cache:
        cached_data, timestamp = weather_cache[cache_key]
        if is_cache_valid(timestamp):
            print(f"✅ Using cached weather data for {cache_key}")
            return cached_data
    
    weather_sources = []
    
    # Define API functions to call concurrently
    api_functions = [
        (get_openmeteo_weather, "Open-Meteo")
    ]
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="weather_api") as executor:
        # Submit all API calls
        future_to_api = {}
        for api_func, api_name in api_functions:
            future = executor.submit(api_func, lat, lon, date)
            future_to_api[future] = api_name
        
        # Collect results as they complete (with timeout)
        for future in as_completed(future_to_api, timeout=20):
            api_name = future_to_api[future]
            try:
                data = future.result(timeout=5)  # Individual timeout
                if data and isinstance(data, dict):
                    weather_sources.append(data)
                    print(f"✅ {api_name} completed successfully")
                else:
                    print(f"⚠️ {api_name} returned invalid data")
            except Exception as e:
                print(f"⚠️ {api_name} failed: {e}")
    
    # Process results
    # Ensure weather_sources is always a valid list
    weather_sources = weather_sources if weather_sources and isinstance(weather_sources, list) else []
    if weather_sources:
        print(f"✅ Weather data collected from {len(weather_sources)} source(s)")
        combined = combine_weather_data(weather_sources)
        today = datetime.today().strftime("%Y-%m-%d")
        season = detect_season(date or today, combined['rainfall'], combined['temp'])
        combined['season'] = season
        
        # Add quality indicators
        combined['source_count'] = len(weather_sources)
        combined['reliability'] = 'high' if len(weather_sources) >= 2 else 'medium'
        
        # Cache the result
        weather_cache[cache_key] = (combined, time.time())
        
        return combined
    else:
        print("⚠️ All weather APIs failed, using geographic fallback")
        fallback_data = get_fallback_weather(lat, lon, date)
        fallback_data['reliability'] = 'low'
        fallback_data['source_count'] = 0
        return fallback_data

# ---------------- Soil APIs and related helpers -----------------
def get_soilgrids_data(lat, lon):
    """Fetch soil texture from SoilGrids REST API (fast) with WMS fallback."""
    soil_data: Dict[str, float] = {}
    sample_points: Dict[str, Tuple[float, float]] = {}
    
    # Try REST API first (much faster)
    print(f"[soilgrids] Attempting REST API for {lat:.4f},{lon:.4f}")
    rest_data = sample_soilgrids_rest_api(lat, lon)
    if rest_data:
        for prop in ["clay", "sand", "silt"]:
            if prop in rest_data:
                value = rest_data[prop].get("value")
                if value is not None:
                    # REST API returns g/kg, convert to percent
                    percent_val = max(0.0, min(100.0, value / 10.0))
                    soil_data[prop] = round(percent_val, 1)
        
        if soil_data:
            print(f"[soilgrids-rest] ✅ Retrieved texture via REST API")
            result: Dict[str, object] = {"source": "SoilGrids REST", "data": soil_data}
            print(f"[soilgrids] texture composition: {soil_data}")
            return result
    
    # Fallback to WMS if REST fails
    print(f"[soilgrids] REST unavailable, falling back to WMS")
    property_layers = {
        "clay": ("/map/clay.map", "clay_0-5cm_mean"),
        "sand": ("/map/sand.map", "sand_0-5cm_mean"),
        "silt": ("/map/silt.map", "silt_0-5cm_mean"),
    }

    for prop, (map_path, layer) in property_layers.items():
        raw_value, sampled_at = sample_soilgrids_wms(map_path, layer, lat, lon)
        if raw_value is None:
            print(f"[soilgrids] no {prop} value available near {lat:.4f},{lon:.4f}")
            continue

        # SoilGrids texture layers are reported in g/kg; convert to percent.
        percent_val = max(0.0, min(100.0, raw_value / 10.0))
        soil_data[prop] = round(percent_val, 1)
        if sampled_at:
            sample_points[prop] = sampled_at

    if not soil_data:
        print("[soilgrids] texture lookup failed for all properties")
        return None

    result: Dict[str, object] = {"source": "SoilGrids WMS", "data": soil_data}
    if sample_points:
        result["sample_points"] = sample_points

    print(f"[soilgrids] texture composition: {soil_data}")
    return result

def get_soil_ph_and_om(lat, lon):
    """Fetch soil pH (H2O) and organic matter from REST API (fast) with WMS fallback."""

    ph_value = None
    om_value = None

    # Try REST API first (includes pH and SOC in single call)
    print(f"[soilgrids] Attempting REST API for pH/SOC at {lat:.4f},{lon:.4f}")
    rest_data = sample_soilgrids_rest_api(lat, lon)
    if rest_data:
        # Extract pH
        if "ph" in rest_data:
            ph_raw = rest_data["ph"].get("value")
            if ph_raw is not None:
                ph_value = round(ph_raw / 10.0, 2)
                print(f"[soilgrids-rest] ✅ pH (0-5cm): {ph_value}")
        
        # Extract SOC and convert to OM
        if "soc" in rest_data:
            soc_raw = rest_data["soc"].get("value")
            if soc_raw is not None:
                percent_carbon = max(0.0, soc_raw * 0.01)  # soc value in dg/kg
                om_calc = percent_carbon * 1.724
                om_value = round(min(max(om_calc, 0.0), 20.0), 2)
                print(f"[soilgrids-rest] ✅ SOC (0-5cm) => OM {om_value}%")
    
    # Fallback to WMS if REST fails or data incomplete
    if ph_value is None:
        print("[soilgrids] REST pH unavailable, trying WMS")
        ph_raw, ph_sample_at = sample_soilgrids_wms('/map/phh2o.map', 'phh2o_0-5cm_mean', lat, lon)
        if ph_raw is not None:
            ph_value = round(ph_raw / 10.0, 2)
            if ph_sample_at:
                print(f"[soilgrids] pH (0-5cm) near {ph_sample_at[0]},{ph_sample_at[1]}: {ph_value}")
        else:
            print("[soilgrids] pH WMS returned no data; using texture-based estimate later")

    if om_value is None:
        print("[soilgrids] REST SOC unavailable, trying WMS")
        soc_raw, soc_sample_at = sample_soilgrids_wms('/map/soc.map', 'soc_0-5cm_mean', lat, lon)
        if soc_raw is not None:
            percent_carbon = max(0.0, soc_raw * 0.01)  # soc pixel_value in dg/kg
            om_calc = percent_carbon * 1.724
            om_value = round(min(max(om_calc, 0.0), 20.0), 2)
            if soc_sample_at:
                print(f"[soilgrids] SOC (0-5cm) near {soc_sample_at[0]},{soc_sample_at[1]} => OM {om_value}%")
        else:
            print("[soilgrids] SOC WMS returned no data for OM estimate")

    return ph_value, om_value

def get_icar_soil_data(lat, lon):
    try:
        print("🇮🇳 Checking ICAR soil classification")
        soil_regions = {
            "alluvial": {
                "bounds": [(22, 31, 75, 88), (8, 12, 76, 80)],
                "composition": {"clay": 30, "sand": 50, "silt": 20}
            },
            "black": {
                "bounds": [(15, 25, 74, 82), (21, 26, 69, 79)],
                "composition": {"clay": 55, "sand": 25, "silt": 20}
            },
            "red": {
                "bounds": [(8, 20, 77, 87), (11, 19, 74, 80)],
                "composition": {"clay": 25, "sand": 60, "silt": 15}
            },
            "laterite": {
                "bounds": [(8, 16, 74, 77), (15, 20, 73, 77)],
                "composition": {"clay": 45, "sand": 40, "silt": 15}
            },
            "desert": {
                "bounds": [(24, 32, 68, 75)],
                "composition": {"clay": 10, "sand": 85, "silt": 5}
            }
        }
        for soil_type, data in soil_regions.items():
            for bounds in data["bounds"]:
                lat_min, lat_max, lon_min, lon_max = bounds
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    print(f"✅ ICAR classification: {soil_type} soil")
                    return {"source": "ICAR-NBSS", "data": data["composition"], "type": soil_type}
        return {"source": "ICAR-NBSS", "data": {"clay": 35, "sand": 45, "silt": 20}, "type": "mixed"}
    except Exception as e:
        print(f"❌ ICAR data failed: {e}")
        raise e

def classify_soil_texture(clay, sand, silt):
    total = clay + sand + silt
    if total == 0:
        return "Unknown"
    clay_pct = (clay / total) * 100
    sand_pct = (sand / total) * 100
    silt_pct = (silt / total) * 100
    if clay_pct >= 40:
        if sand_pct <= 45:
            return "Clay"
        else:
            return "Sandy Clay"
    elif clay_pct >= 27:
        if sand_pct <= 20:
            return "Silty Clay"
        elif sand_pct <= 45:
            return "Clay Loam"
        else:
            return "Sandy Clay Loam"
    elif clay_pct >= 12:
        if silt_pct >= 50:
            return "Silt Loam"
        elif sand_pct >= 52:
            return "Sandy Loam"
        else:
            return "Loam"
    else:
        if silt_pct >= 50:
            if silt_pct >= 80:
                return "Silt"
            else:
                return "Silt Loam"
        elif sand_pct >= 85:
            return "Sandy"
        elif sand_pct >= 70:
            return "Loamy Sand"
        else:
            return "Loam"

def combine_soil_data(soil_sources):
    if not soil_sources:
        raise Exception("No soil data available")
    
    # Filter out None values and ensure we only process valid dictionaries
    valid_sources = [s for s in soil_sources if s is not None and isinstance(s, dict)]
    
    compositions = [s['data'] for s in valid_sources if 'data' in s and s['data'] is not None]
    sources = [s.get('source', 'Unknown') for s in valid_sources]
    
    if not compositions:
        raise Exception("No valid soil composition data")
    
    # Ensure compositions is a valid list before len() operations
    compositions = compositions if compositions and isinstance(compositions, list) else []
    if len(compositions) == 0:
        raise Exception("No valid soil composition data")
        
    avg_clay = sum(c.get('clay', 0) for c in compositions if c and isinstance(c, dict)) / len(compositions)
    avg_sand = sum(c.get('sand', 0) for c in compositions if c and isinstance(c, dict)) / len(compositions)
    avg_silt = sum(c.get('silt', 0) for c in compositions if c and isinstance(c, dict)) / len(compositions)
    soil_type = classify_soil_texture(avg_clay, avg_sand, avg_silt)
    print(f"🌍 Combined soil: {soil_type} (Clay: {avg_clay:.1f}%, Sand: {avg_sand:.1f}%, Silt: {avg_silt:.1f}%)")
    print(f"📊 Sources: {', '.join(sources)}")
    return {
        "soil_type": soil_type,
        "composition": {
            "clay": round(avg_clay, 1),
            "sand": round(avg_sand, 1),
            "silt": round(avg_silt, 1)
        },
        "sources": sources
    }

def get_soil_data(lat, lon):
    """Enhanced soil data fetching with caching and concurrent API calls"""
    # Check cache first
    cache_key = get_cache_key(lat, lon)
    if cache_key in soil_cache:
        cached_data, timestamp = soil_cache[cache_key]
        if is_cache_valid(timestamp):
            print(f"✅ Using cached soil data for {cache_key}")
            return cached_data
    
    soil_sources = []
    
    # Define API functions for concurrent soil data fetching
    api_functions = [
        (get_soilgrids_data, "SoilGrids"),
    ]
    
    # Add ICAR for India region
    if 8 <= lat <= 37 and 68 <= lon <= 97:
        api_functions.append((get_icar_soil_data, "ICAR"))
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="soil_main") as executor:
        future_to_api = {}
        for api_func, api_name in api_functions:
            future = executor.submit(api_func, lat, lon)
            future_to_api[future] = api_name
        
        # Collect results as they complete
        try:
            for future in as_completed(future_to_api, timeout=15):
                api_name = future_to_api[future]
                try:
                    soil_data = future.result(timeout=5)
                    if soil_data and isinstance(soil_data, dict) and 'data' in soil_data:
                        soil_sources.append(soil_data)
                        print(f"✅ {api_name} completed successfully")
                    else:
                        print(f"⚠️ {api_name} returned invalid data")
                except Exception as e:
                    print(f"⚠️ {api_name} failed: {e}")
        except Exception:
            # Timeout or other error - cancel remaining futures
            for future in future_to_api:
                future.cancel()
    
    # Process soil data
    # Ensure soil_sources is always a valid list
    soil_sources = soil_sources if soil_sources and isinstance(soil_sources, list) else []
    if soil_sources:
        try:
            combined_data = combine_soil_data(soil_sources)
            combined_data['source_count'] = len(soil_sources)
            combined_data['reliability'] = 'high' if len(soil_sources) >= 2 else 'medium'

            comp = combined_data.get('composition', {}) if isinstance(combined_data, dict) else {}
            chem_guess = estimate_soil_chemistry(
                combined_data.get('soil_type'),
                comp.get('clay'),
                comp.get('sand'),
                comp.get('silt')
            )

            ph_value, om_value = get_soil_ph_and_om(lat, lon)
            if ph_value is None or not isinstance(ph_value, (int, float)):
                ph_value = chem_guess.get('ph')
            combined_data['ph'] = round(float(ph_value), 2) if ph_value is not None else chem_guess.get('ph', 6.6)

            if om_value is None or not isinstance(om_value, (int, float)):
                om_value = chem_guess.get('om')
            if om_value is not None:
                combined_data['organic_matter'] = f"{float(om_value):.1f}%"
                combined_data['organic_matter_pct'] = round(float(om_value), 1)
            else:
                combined_data['organic_matter'] = "2.8%"
                combined_data['organic_matter_pct'] = 2.8
            
            # Cache the result
            soil_cache[cache_key] = (combined_data, time.time())
            print(f"✅ Soil data combined from {len(soil_sources)} source(s)")
            
            return combined_data
        except Exception as e:
            print(f"⚠️ Failed to combine soil data: {e}")
            fallback_data = get_fallback_soil(lat, lon)
            fallback_data['reliability'] = 'low'
            fallback_data['source_count'] = 0
            comp = fallback_data.get('composition', {})
            chem_guess = estimate_soil_chemistry(
                fallback_data.get('soil_type'),
                comp.get('clay'),
                comp.get('sand'),
                comp.get('silt')
            )
            ph_value, om_value = get_soil_ph_and_om(lat, lon)
            if ph_value is None:
                ph_value = chem_guess.get('ph')
            if ph_value is not None:
                fallback_data['ph'] = round(float(ph_value), 2)
            if om_value is None:
                om_value = chem_guess.get('om')
            if om_value is not None:
                fallback_data['organic_matter'] = f"{float(om_value):.1f}%"
                fallback_data['organic_matter_pct'] = round(float(om_value), 1)
            soil_cache[cache_key] = (fallback_data, time.time())
            return fallback_data
    else:
        print("⚠️ All soil APIs failed, using geographic estimation")
        fallback_data = get_fallback_soil(lat, lon)
        fallback_data['reliability'] = 'low'
        fallback_data['source_count'] = 0
        comp = fallback_data.get('composition', {})
        chem_guess = estimate_soil_chemistry(
            fallback_data.get('soil_type'),
            comp.get('clay'),
            comp.get('sand'),
            comp.get('silt')
        )
        ph_value, om_value = get_soil_ph_and_om(lat, lon)
        if ph_value is None:
            ph_value = chem_guess.get('ph')
        if ph_value is not None:
            fallback_data['ph'] = round(float(ph_value), 2)
        if om_value is None:
            om_value = chem_guess.get('om')
        if om_value is not None:
            fallback_data['organic_matter'] = f"{float(om_value):.1f}%"
            fallback_data['organic_matter_pct'] = round(float(om_value), 1)
        soil_cache[cache_key] = (fallback_data, time.time())
        return fallback_data

def detect_season(date_str, rainfall, temp):
    month = datetime.strptime(date_str, "%Y-%m-%d").month if date_str else datetime.today().month
    if month in [6, 7, 8, 9]:
        return "Kharif"
    elif month in [10, 11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"

def get_fallback_weather(lat, lon, date=None):
    if 8 <= lat <= 37 and 68 <= lon <= 97:
        if lat >= 28:
            base_temp = 24
        elif lat <= 15:
            base_temp = 28
        else:
            base_temp = 26
    else:
        base_temp = 25
    month = datetime.strptime(date, "%Y-%m-%d").month if date else datetime.today().month
    if month in [4, 5, 6]:
        temp = base_temp + 5
        rainfall = 10
    elif month in [7, 8, 9]:
        temp = base_temp
        rainfall = 150
    elif month in [10, 11]:
        temp = base_temp - 2
        rainfall = 30
    else:
        temp = base_temp - 8
        rainfall = 15
    season = detect_season(date or datetime.today().strftime("%Y-%m-%d"), rainfall, temp)
    weather_fallback = {
        "temp": temp,
        "rainfall": rainfall,
        "season": season,
        "sources": ["Geographic Estimation"]
    }
    print(f"🏞️ Using geographic fallback weather for lat={lat}, lon={lon}")
    print(f"🎯 Fallback weather data: {weather_fallback}")
    return weather_fallback

def get_fallback_soil(lat, lon):
    """
    Provides default soil data based on geographic location when APIs fail
    """
    print(f"🏞️ Using geographic fallback for lat={lat}, lon={lon}")
    
    # Default composition values
    default_clay = 30.0
    default_sand = 40.0
    default_silt = 30.0
    
    if 8 <= lat <= 37 and 68 <= lon <= 97:  # India region
        if ((22 <= lat <= 31) and (75 <= lon <= 88)):  # Gangetic plains
            soil_type = "Alluvial"
            default_clay = 25.0
            default_sand = 45.0
            default_silt = 30.0
        elif ((15 <= lat <= 25) and (74 <= lon <= 82)):  # Deccan plateau
            soil_type = "Clay"
            default_clay = 45.0
            default_sand = 30.0
            default_silt = 25.0
        elif ((24 <= lat <= 32) and (68 <= lon <= 75)):  # Western region
            soil_type = "Sandy"
            default_clay = 15.0
            default_sand = 60.0
            default_silt = 25.0
        elif lat <= 15:  # Southern India
            soil_type = "Loam"
            default_clay = 30.0
            default_sand = 40.0
            default_silt = 30.0
        else:  # Other regions in India
            soil_type = "Clay Loam"
            default_clay = 35.0
            default_sand = 35.0
            default_silt = 30.0
    else:  # Non-India regions
        soil_type = "Loam"
        default_clay = 30.0
        default_sand = 40.0
        default_silt = 30.0
    
    chem_guess = estimate_soil_chemistry(soil_type, default_clay, default_sand, default_silt)

    fallback_data = {
        "soil_type": soil_type,
        "composition": {
            "clay": default_clay,
            "sand": default_sand,
            "silt": default_silt
        },
        "sources": ["Geographic Estimation"],
        "note": "Fallback data used due to API failures",
        "ph": chem_guess.get("ph", 6.6),
        "organic_matter": f"{chem_guess.get('om', 2.3):.1f}%",
        "organic_matter_pct": round(chem_guess.get('om', 2.3), 1)
    }
    
    print(f"🎯 Fallback soil data: {soil_type} (Clay: {default_clay}%, Sand: {default_sand}%, Silt: {default_silt}%)")
    return fallback_data

    # Import our new crop advisory module
from crop_advisory import load_impact_data, calculate_soil_health_score, calculate_rainfall_impact

MODEL_DIR = os.getenv("MODEL_DIR", "models")
USE_ML = os.getenv("USE_ML", "true").lower() != "false"
ML_ALPHA = float(os.getenv("ML_ALPHA", "0.2"))
ML_BLEND_THRESHOLD = float(os.getenv("ML_BLEND_THRESHOLD", "15.0"))

# Weight-based scoring for structured decision logic
WEIGHTS = {
    "temp": 0.2,
    "rainfall": 0.2,
    "soil": 0.3,
    "season": 0.2,
    "rotation": 0.1
}

_env_model = None
_features = None
_crop_names = None
_model_lock = threading.Lock()
_model_loaded_time = None
_model_info = {}


def _load_models():
    global _env_model, _features, _crop_names, _model_loaded_time, _model_info
    with _model_lock:
        if _env_model is not None:
            return
        try:
            _env_model = joblib.load(os.path.join(MODEL_DIR, "env_model.joblib"))
            _features = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
            _crop_names = joblib.load(os.path.join(MODEL_DIR, "crop_names.joblib"))
            _model_loaded_time = time.time()
            feat_count = len(_features) if isinstance(_features, (list, tuple)) else 0
            crop_count = len(_crop_names) if isinstance(_crop_names, (list, tuple)) else 0
            _model_info = {
                "env_model": True,
                "feature_count": feat_count,
                "crop_count": crop_count,
                "features_sample": (_features[:30] if isinstance(_features, (list, tuple)) else [])
            }
        except Exception:
            _env_model = None
            _features = None
            _crop_names = None
            _model_loaded_time = None
            _model_info = {"env_model": False, "feature_count": 0, "crop_count": 0, "features_sample": []}

def _normalise_feature_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned.lower().replace('/', '_').replace(' ', '_')


def _to_float(value, default: float) -> float:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_environment_vector(soil_data, rainfall, temperature, season):
    if not isinstance(_features, (list, tuple)):
        return None

    row = {feature: 0.0 for feature in _features}

    row_key = 'rainfall'
    if row_key in row:
        row[row_key] = _to_float(rainfall, 0.0)
    row_key = 'temperature'
    if row_key in row:
        row[row_key] = _to_float(temperature, 0.0)
    row_key = 'ph'
    if row_key in row:
        soil_ph = soil_data.get('ph') if isinstance(soil_data, dict) else None
        row[row_key] = _to_float(soil_ph, 6.5)

    soil_label = None
    if isinstance(soil_data, dict):
        soil_label = soil_data.get('soil_type') or soil_data.get('texture')
    soil_norm = _normalise_feature_label(soil_label)
    if soil_norm:
        soil_col = f'soil__{soil_norm}'
        if soil_col in row:
            row[soil_col] = 1.0
    elif 'soil__loam' in row:
        row['soil__loam'] = 1.0

    season_norm = _normalise_feature_label(season)
    if season_norm:
        season_col = f'season__{season_norm}'
        if season_col in row:
            row[season_col] = 1.0
    elif 'season__year-round' in row:
        row['season__year-round'] = 1.0

    try:
        return pd.DataFrame([row], columns=_features)
    except Exception:
        return None


def predict_all_ml_scores(soil_data, rainfall, temperature, season=None):
    if not USE_ML:
        return {}
    _load_models()
    if _env_model is None or not isinstance(_crop_names, (list, tuple)):
        return {}
    Xv = _build_environment_vector(soil_data, rainfall, temperature, season)
    if Xv is None:
        return {}
    try:
        preds = _env_model.predict(Xv.values)
        if preds is None or len(preds) == 0:
            return {}
        row = np.asarray(preds[0]).ravel()
        results = {}
        for idx, crop_name in enumerate(_crop_names):
            if idx < len(row):
                results[crop_name] = float(max(0.0, min(100.0, row[idx])))
        return results
    except Exception:
        return {}


def predict_ml_score(crop_name, soil_data, rainfall, temperature, season=None):
    scores = predict_all_ml_scores(soil_data, rainfall, temperature, season)
    return scores.get(crop_name)


@app.route('/ml-health')
def ml_health():
    """Return ML model load status and basic metadata for monitoring."""
    try:
        _load_models()
        loaded = (_env_model is not None)
        files = {}
        for name in ['env_model.joblib', 'features.joblib', 'crop_names.joblib']:
            path = os.path.join(MODEL_DIR, name)
            if os.path.exists(path):
                try:
                    files[name] = {'exists': True, 'mtime': os.path.getmtime(path)}
                except Exception:
                    files[name] = {'exists': True, 'mtime': None}
            else:
                files[name] = {'exists': False, 'mtime': None}

        return jsonify({
            'use_ml': USE_ML,
            'loaded': loaded,
            'model_info': _model_info,
            'model_loaded_time': _model_loaded_time,
            'model_files': files,
            'ml_alpha': ML_ALPHA,
            'ml_blend_threshold': ML_BLEND_THRESHOLD
        })
    except Exception as e:
        return jsonify({'use_ml': USE_ML, 'loaded': False, 'error': str(e)}), 500

# NOTE: This is a scaffold-only integration. To use ML in recommendation logic,
# call `predict_ml_score(...)` where you compute the rule-based score (e.g. inside
# `recommend_crop_full`) and blend as:
#   ml = predict_ml_score(crop_name, soil_data, rainfall, temperature)
#   if ml is not None:
#       combined_score = ML_ALPHA * ml + (1 - ML_ALPHA) * rule_score
#   else:
#       combined_score = rule_score


# Build a human-readable explanation block for a crop recommendation
def build_crop_explanation(rec, soil_data, weather, past_crop):
    """Create a clean, structured explanation with no repetitions."""
    try:
        if not rec or not isinstance(rec, dict):
            return ""

        soil_data = soil_data or {}
        weather = weather or {}
        past_crop = past_crop or ""
        matches = rec.get('matches') if isinstance(rec.get('matches'), dict) else {}

        def _safe_float(value):
            try:
                if value is None:
                    return None
                if isinstance(value, str) and not value.strip():
                    return None
                result = float(value)
                if math.isnan(result):
                    return None
                return result
            except (ValueError, TypeError):
                return None

        def _match_positive(key):
            value = matches.get(key)
            if isinstance(value, list):
                return any(bool(v) if isinstance(v, bool) else (isinstance(v, (int, float)) and v > 0) for v in value)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value > 0
            return False

        def _match_negative(key):
            value = matches.get(key)
            if isinstance(value, list):
                return any((isinstance(v, bool) and not v) or (isinstance(v, (int, float)) and v < 0) for v in value)
            if isinstance(value, bool):
                return not value
            if isinstance(value, (int, float)):
                return value < 0
            return False

        crop_name = rec.get('crop') or rec.get('Crop') or 'Top crop'
        soil_type = soil_data.get('soil_type') if isinstance(soil_data, dict) else None
        weather_temp = _safe_float(weather.get('temp'))
        weather_rain = _safe_float(weather.get('rainfall'))

        # ========== WHY THIS CROP ==========
        why_lines = ["🌱 WHY THIS CROP?"]
        
        # Best match reason
        why_lines.append(f"✓ Best match for soil and season.")
        
        # Temp match
        if _match_positive('temp'):
            why_lines.append(f"✓ Temperature suitable ({int(weather_temp)}°C expected).")
        
        # Rainfall match
        if _match_positive('rainfall'):
            why_lines.append(f"✓ Rainfall pattern fits this crop.")

        # ========== WHAT YOU SHOULD DO ==========
        actions = set()
        
        if _match_negative('rainfall') and weather_rain is not None:
            actions.add("Plan irrigation—rainfall is below ideal.")
        if _match_negative('soil') and soil_type:
            actions.add(f"Add compost to prepare your {soil_type} soil.")
        if _match_negative('temp'):
            actions.add("Use shade or mulch for temperature protection.")
        if weather_rain and weather_rain > 100:
            actions.add("Prepare drainage—rainfall looks heavy.")
        
        if not actions:
            actions.add("Follow standard field preparation.")

        action_lines = ["", "📋 WHAT YOU SHOULD DO?"]
        for action in sorted(actions):
            action_lines.append(f"• {action}")

        # ========== RISKS ==========
        risks = set()
        
        if _match_negative('temp') and weather_temp is not None:
            risks.add(f"Temperature outside ideal range ({int(weather_temp)}°C).")
        
        if _match_negative('rainfall') and weather_rain is not None:
            risks.add(f"Rainfall mismatch ({int(weather_rain)} mm expected).")
        
        if _match_negative('soil') and soil_type:
            risks.add(f"Your {soil_type} soil needs improvement.")
        
        if _match_negative('counter_crop') and past_crop:
            risks.add(f"Watch for pests carried from {past_crop}.")
        
        soil_details = rec.get('soil_health_details') if isinstance(rec.get('soil_health_details'), dict) else {}
        soil_risk_level = soil_details.get('risk_level') if isinstance(soil_details, dict) else None
        if soil_risk_level and soil_risk_level not in ("Highly Recommended", "Excellent Match"):
            risks.add("Soil health needs improvement before sowing.")
        
        if not risks:
            risks.add("Keep weekly field walks to monitor growth.")

        risk_lines = ["", "⚠️ RISKS"]
        for i, risk in enumerate(sorted(risks), 1):
            if i <= 3:
                risk_lines.append(f"• {risk}")

        # Combine all sections
        all_lines = why_lines + action_lines + risk_lines
        return "\n".join(all_lines).strip()

    except Exception as exc:
        print(f"⚠️ Failed to build explanation: {exc}")
        return ""


def generate_crop_insights(rec, crop_info, weather_data, soil_data):
    explanation = []
    action_plan = []
    risks = []

    crop_info = crop_info or {}
    weather_data = weather_data or {}
    soil_data = soil_data or {}

    # -------- BASIC DATA --------
    crop_name = rec.get("crop")
    temp = weather_data.get("temp")
    rain = weather_data.get("rainfall")
    soil_type = soil_data.get("soil_type")

    # Crop ideal ranges
    temp_range = crop_info.get("Temp")
    if not temp_range and crop_info.get("Temp_Min") is not None and crop_info.get("Temp_Max") is not None:
        temp_range = f"{crop_info.get('Temp_Min')}-{crop_info.get('Temp_Max')}"
    temp_range = temp_range or "0-0"

    rain_range = crop_info.get("Rain")
    if not rain_range and crop_info.get("Rain_Min") is not None and crop_info.get("Rain_Max") is not None:
        rain_range = f"{crop_info.get('Rain_Min')}-{crop_info.get('Rain_Max')}"
    rain_range = rain_range or "0-0"

    try:
        t_min, t_max = map(int, str(temp_range).split("-"))
        r_min, r_max = map(int, str(rain_range).split("-"))
    except Exception:
        t_min, t_max = 0, 50
        r_min, r_max = 0, 2000

    # -------- EXPLANATION --------

    # Temperature
    if temp is not None:
        if t_min <= temp <= t_max:
            explanation.append(
                f"Temperature ({temp}°C) is within ideal range ({t_min}-{t_max}°C)"
            )
        elif temp > t_max:
            explanation.append(
                f"High temperature ({temp}°C) slightly exceeds ideal range ({t_min}-{t_max}°C)"
            )
        else:
            explanation.append(
                f"Low temperature ({temp}°C) below ideal range ({t_min}-{t_max}°C)"
            )

    # Rainfall
    if rain is not None:
        if r_min <= rain <= r_max:
            explanation.append(
                f"Rainfall ({rain} mm) is within suitable range ({r_min}-{r_max} mm)"
            )
        elif rain < r_min:
            explanation.append(
                f"Low rainfall ({rain} mm) below requirement ({r_min}-{r_max} mm)"
            )
        else:
            explanation.append(
                f"High rainfall ({rain} mm) exceeds requirement ({r_min}-{r_max} mm)"
            )

    # Soil
    explanation.append(f"Compatible with {soil_type} soil")

    # Crop nature hint
    if rain is not None and rain < 50:
        explanation.append(f"{crop_name} is suitable for low water conditions")

    # -------- ACTION PLAN --------

    if rain is not None and rain < r_min:
        action_plan.append("Provide irrigation support (drip/sprinkler recommended)")

    if soil_data.get("soil_health", {}).get("score", 100) < 60:
        action_plan.append("Improve soil with compost or organic manure")

    if temp is not None and temp > t_max:
        action_plan.append("Use mulching to retain soil moisture")

    action_plan.append("Follow proper sowing schedule and spacing")

    # -------- RISKS --------

    if temp is not None and temp > t_max:
        risks.append("High temperature may cause heat stress")

    if rain is not None and rain < r_min:
        risks.append("Low rainfall may affect crop growth")

    if soil_data.get("organic_matter_pct", 2) < 1.5:
        risks.append("Low organic matter may reduce soil fertility")

    return {
        "explanation": explanation,
        "action_plan": action_plan,
        "risks": risks
    }

# ---------------- FIXED recommend_crop_full -----------------
def recommend_crop_full(soil_data, past_crop=None, weather=None, show_all=False):
    # Map internal labels to frontend labels used by the UI
    label_map = {
        "Highly Recommended": "Excellent Match",
        "Recommended with Caution": "Good Match",
        "Not Recommended": "Fair Match",
        "No Impact Data": "Unknown Impact",
        "Unknown": "Unknown Impact"
    }

    recommendations = []
    past_counters = []
    
    # Load crop impacts data
    try:
        impact_data = load_impact_data()
    except Exception as e:
        print(f"⚠️ Failed to load crop impacts data: {e}")
        impact_data = {"crop_impacts": {}}
    
    # Ensure inputs are safe
    if not soil_data or not isinstance(soil_data, dict):
        soil_data = {}
    if not crop_data or not isinstance(crop_data, list):
        return []

    # Handle past crop
    if past_crop and isinstance(past_crop, str):
        past_crop_meta = None
        for c in crop_data:
            if not c or not isinstance(c, dict):
                continue
            crop_name = c.get('Crop')
            if crop_name and isinstance(crop_name, str) and crop_name.lower() == past_crop.lower():
                past_counters = c.get('Counters') or []
                past_crop_meta = c
                if not isinstance(past_counters, list):
                    past_counters = []
                break

    # Handle soil type safely
    soil_type = soil_data.get('soil_type')
    if not soil_type:
        print("⚠️ Soil type missing, skipping crop filtering")
        soil_type = ""

    rainfall_for_ml = weather.get('rainfall') if isinstance(weather, dict) else None
    temperature_for_ml = weather.get('temp') if isinstance(weather, dict) else None
    season_for_ml = weather.get('season') if isinstance(weather, dict) else None
    if not season_for_ml:
        season_for_ml = 'Year-round'
    ml_predictions = {}
    if USE_ML:
        try:
            ml_predictions = predict_all_ml_scores(soil_data, rainfall_for_ml, temperature_for_ml, season_for_ml)
        except Exception:
            ml_predictions = {}

    for crop in crop_data:
        if not crop or not isinstance(crop, dict):
            continue

        # scoring trace variables for debugging (reset per crop)
        initial_score = 100.0  # Start with perfect score
        matches = {}
        penalties = []

        # ==============================
        # 🔹 DELEGATE SCORING TO score_engine
        # ==============================
        user_env_for_scoring = {
            "temp": weather.get("temp") if weather else None,
            "rainfall": weather.get("rainfall") if weather else None,
            "season": weather.get("season") if weather else None,
            "Soil_Type": soil_data.get("soil_type") if soil_data else None,
            "pH": soil_data.get("ph") if soil_data else None,
            "organic_matter": soil_data.get("organic_matter_pct") if soil_data else None,
            "temp_anomalies": weather.get("temp_anomalies") if weather else False,
            "rain_anomalies": weather.get("rain_anomalies") if weather else False
        }
        
        rule_result = score_crop(crop, user_env_for_scoring)
        rule_score_raw = float(rule_result.get("rule_score", 50))
        soil_health_score = rule_result.get("soil_score", 50)
        penalties = rule_result.get("penalties", [])

        # Get temperature and rainfall ranges from crop data for matches tracking
        crop_temp_min = crop.get('Temp_Min')
        crop_temp_max = crop.get('Temp_Max')
        crop_rain_min = crop.get('Rain_Min')
        crop_rain_max = crop.get('Rain_Max')
        
        # Build matches for UI/debugging
        if weather is not None:
            current_temp = weather.get("temp")
            if current_temp is not None and crop_temp_min is not None and crop_temp_max is not None:
                matches["temp"] = (crop_temp_min <= current_temp <= crop_temp_max)

        if weather is not None:
            current_rain = weather.get("rainfall")
            if current_rain is not None and crop_rain_min is not None and crop_rain_max is not None:
                matches["rainfall"] = (crop_rain_min <= current_rain <= crop_rain_max)

        # Soil type match
        soil_types = crop.get('Soil_Type', [])
        if soil_data and soil_data.get('soil_type') and soil_types:
            matches['soil'] = soil_data.get('soil_type') in soil_types
        else:
            matches['soil'] = True

        # Season match
        if weather and weather.get('season'):
            current_season = weather.get('season')
            crop_seasons = crop.get('Season', [])
            matches['season'] = current_season in crop_seasons
        else:
            matches['season'] = True

        # Rotation match
        matches['counter_crop'] = not (past_crop and past_crop in (crop.get('Counters') or []))

        # Soil health and rainfall impact (for detailed data)
        crop_impact_data = impact_data.get('crop_impacts', {}).get(crop.get('Crop', ''), {})
        if crop_impact_data:
            soil_health = calculate_soil_health_score(soil_data, crop_impact_data)
            soil_health_orig = soil_health.get('recommendation')
            matches['soil_health'] = [soil_health_orig == "Highly Recommended"]
        else:
            soil_health = {"score": 50, "impacts": [], "risk_level": "Unknown", "recommendation": "No Impact Data"}
            soil_health_orig = soil_health.get('recommendation')
            matches['soil_health'] = [False]
            
        # Calculate rainfall impact
        if weather and weather.get('rainfall') is not None:
            rainfall_impact = calculate_rainfall_impact(
                weather['rainfall'],
                crop.get('Crop', ''),
                impact_data
            )
            rainfall_score = rainfall_impact.get('score')
            rainfall_orig = rainfall_impact.get('recommendation')
            # Just store values, don't average
            penalties.append({
                "reason": f"Rainfall Impact - {rainfall_impact.get('severity')}",
                "penalty": -rainfall_impact.get('penalty', 0)
            })
            matches['rainfall_impact'] = [rainfall_orig == "Highly Recommended"]
        
        # ==============================
        # 🔹 FINALIZE SCORES WITH PROPER HYBRID LOGIC
        # ==============================
        rule_score_raw = float(rule_score_raw)  # Already from score_crop()
        rule_score = round(rule_score_raw)

        # Get ML score
        crop_name = crop.get('Crop', '')
        ml_score = ml_predictions.get(crop_name) if ml_predictions else None
        
        # Get rotation score
        rotation_score = get_rotation_score(crop, past_crop, pd.DataFrame(crop_data))
        
        # Ensure all scores are in 0-100 range
        rule_score_val = float(np.clip(rule_score_raw, 0, 100))
        ml_score_val = float(np.clip(ml_score, 0, 100)) if ml_score is not None else 50.0
        soil_health_val = float(np.clip(soil_health_score, 0, 100)) if soil_health_score is not None else 50.0
        rotation_score_val = float(np.clip(rotation_score, 0, 100)) if rotation_score is not None else 50.0

        # ==============================
        # 🔹 HYBRID FINAL SCORE via engine
        # ==============================
        final_score, _ = compute_final_score(
            rule_score_val, ml_score_val, soil_health_val, rotation_score_val,
            user_env=user_env_for_scoring
        )
        
        print("🔥 FINAL SCORE:", final_score, "ML:", ml_score_val, "RULE:", rule_score_val)

        # Debug logging deferred until top recommendations are finalized

        # Store matches as simple values (not wrapped in lists)
        processed_matches = {}
        for key, value in matches.items():
            processed_matches[key] = value

        # map backend recommendation strings to frontend labels
        mapped_soil_label = label_map.get(soil_health_orig, soil_health_orig) if soil_health_orig else ("Unknown Impact" if crop_impact_data else "Unknown Impact")
        if 'rainfall_impact' in locals() and isinstance(rainfall_impact, dict):
            rainfall_impact['mapped_recommendation'] = label_map.get(rainfall_orig, rainfall_orig)

        # The backend no longer computes per-recommendation 'why_recommended' or
        # 'possible_limitations' lists. Explanations are generated via
        # `build_crop_explanation` and attached to recommendations as
        # `explanation` strings. This avoids sending those separate fields.

        # Rotation benefit sourced from current crop metadata
        try:
            rb_source = crop.get('Rotation_Benefit') or crop.get('RotationBenefits')
            if isinstance(rb_source, list):
                rotation_benefit = ", ".join([str(item).strip() for item in rb_source if str(item).strip()])
            else:
                rotation_benefit = str(rb_source).strip() if rb_source else ""
            if not rotation_benefit:
                rotation_benefit = "Supports soil recovery and breaks pest cycles"
        except Exception:
            rotation_benefit = "Supports soil recovery and breaks pest cycles"

        # Build a compact rotation_insights string for this recommendation
        try:
            prev_crop = past_crop_meta.get('Crop') if past_crop_meta and isinstance(past_crop_meta, dict) else (past_crop or 'Unknown')
            def _normalize(value):
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
                if isinstance(value, str):
                    return [part.strip() for part in value.replace('\n', ';').split(';') if part.strip()]
                return []

            past_effects = []
            if past_crop_meta and isinstance(past_crop_meta, dict):
                for key in ('Side_Effect', 'Side_Effects', 'Soil_Effect', 'Soil_Effects'):
                    past_effects.extend(_normalize(past_crop_meta.get(key)))
                past_effects.extend(_normalize(past_crop_meta.get('Counters')))
            past_effects = [item for item in dict.fromkeys(past_effects)]
            past_effects_text = ", ".join(past_effects) if past_effects else "None recorded"

            counters_val = crop.get('Counters')
            current_counters = _normalize(counters_val)
            counters_text = ", ".join(dict.fromkeys(current_counters)) if current_counters else "None recorded"

            rotation_insights = (
                f"🟢 Previous crop: {prev_crop}\n"
                f"⚠️ Problems caused by past crop: {past_effects_text}\n"
                f"🌱 How the recommended crop helps recover: {rotation_benefit}\n"
                f"🔎 Remaining risks for this new crop: {counters_text}"
            )
        except Exception:
            rotation_insights = (
                f"🟢 Previous crop: {past_crop or 'Unknown'}\n"
                f"⚠️ Problems caused by past crop: None recorded\n"
                f"🌱 How the recommended crop helps recover: Supports soil recovery and breaks pest cycles\n"
                f"🔎 Remaining risks for this new crop: None recorded"
            )

        result = {
            "crop": crop.get('Crop', 'Unknown'),
            "season": crop.get('Season', ['Unknown']),
            # Final hybrid score (single canonical score key)
            "final_score": final_score,
            "soil_health_impact": mapped_soil_label,
            "soil_health_original": soil_health_orig,
            "soil_health_details": {
                "score": soil_health['score'],
                "risk_level": soil_health['risk_level'],
                "impacts": soil_health['impacts']
            } if crop_impact_data else {"score": 50, "risk_level": "Unknown", "impacts": []},
            "rainfall_impact": rainfall_impact if 'rainfall_impact' in locals() else {
                "score": 50,
                "severity": "Unknown",
                "risk_level": "Unknown",
                "recommendation": "No Impact Data"
            },
            "temp_range": f"{crop_temp_min}-{crop_temp_max}" if crop_temp_min is not None and crop_temp_max is not None else "N/A",
            "rainfall_range": f"{crop_rain_min}-{crop_rain_max}" if crop_rain_min is not None and crop_rain_max is not None else "N/A",
            "soil_types": soil_types,
            "rotation_benefit": rotation_benefit,
            "rotation_insights": rotation_insights,
            "matches": processed_matches,
            "penalties": penalties,
            "Legume": bool(crop.get('Legume')),
            "Improves_OM": bool(crop.get('Improves_OM')),
            "Deep_Roots": bool(crop.get('Deep_Roots')),
            "PestBreak": bool(crop.get('PestBreak')),
            "Heavy_Nutrient_Depletion": bool(crop.get('Heavy_Nutrient_Depletion')),
            "Reduces_OM": bool(crop.get('Reduces_OM')),
            "PestProne": bool(crop.get('PestProne')),
            "Compaction": bool(crop.get('Compaction')),
            "impact_data": {
                "nutrient_depletion": crop_impact_data.get('nutrient_depletion', {}),
                "disease_risk": crop_impact_data.get('disease_risk', {}),
                "physical_degradation": crop_impact_data.get('physical_degradation', {}),
                "allelopathy": crop_impact_data.get('allelopathy', False),
                "irrigation_efficiency": crop_impact_data.get('irrigation_efficiency', 0.75)
            } if crop_impact_data else {},
            "debug": {
                "rule_score": rule_score_val,
                "ml_score": ml_score_val if ml_score is not None else None,
                "soil_health_score": soil_health_val,
                "rotation_score": rotation_score_val,
                "final_score_raw": final_score,
                "hybrid_weights": {
                    "ml": 0.4,
                    "rule": 0.3,
                    "soil": 0.2,
                    "rotation": 0.1
                }
            },
            # Detailed score breakdown for frontend/research
            "ml_score": ml_score_val if ml_score is not None else None,
            "rule_score": rule_score_val,
            "soil_health_score": soil_health_val,
            "rotation_score": rotation_score_val
        }

        # Add confidence categorization
        confidence = "High" if final_score > 75 else "Medium" if final_score > 50 else "Low"
        result["confidence"] = confidence

        # Add recommendation type indicator
        recommendation_type = "Hybrid" if ml_score is not None else "Rule-Based"
        result["recommendation_type"] = recommendation_type

        # Do NOT attach per-recommendation explanation here; top-crop explanation
        # will be generated once recommendations are sorted and the top item is known.

        recommendations.append(result)

    # Sort recommendations by score and soil health impact
    # Sort by raw final score (descending) to preserve tiny differences, then by mapped soil health label
    def _sort_key(x):
        # Primary: use debug.final_score_raw when available, else fall back to rounded final score
        primary = None
        try:
            primary = float(x.get('debug', {}).get('final_score_raw', x.get('final_score', 0)))
        except Exception:
            primary = float(x.get('final_score', 0))

        # Secondary: priority mapping for soil health labels
        secondary = 0
        if x.get('soil_health_impact') == "Excellent Match":
            secondary = 2
        elif x.get('soil_health_impact') == "Good Match":
            secondary = 1
        else:
            secondary = 0

        return (-primary, -secondary)

    recommendations.sort(key=_sort_key)
    
    # ==============================
    # 🔹 GENERATE ACTION PLAN & REJECTION REASONS
    # ==============================
    # Action plan based on current conditions
    action_plan = []
    if weather and weather.get('rainfall') and weather.get('rainfall') < 400:
        action_plan.append("Irrigation planning required - rainfall is below optimal (< 400mm)")
    if weather and weather.get('temp_anomalies'):
        action_plan.append("Temperature fluctuations detected - monitor crop closely and use greenhouse if needed")
    if weather and weather.get('rain_anomalies'):
        action_plan.append("Unusual rainfall patterns - ensure drainage and irrigation backup ready")
    if soil_data and soil_data.get('ph') and (soil_data.get('ph') < 6.0 or soil_data.get('ph') > 7.5):
        action_plan.append(f"Soil pH adjustment needed (current: {soil_data.get('ph')})")
    if soil_data and soil_data.get('organic_matter_pct') and soil_data.get('organic_matter_pct') < 2.0:
        action_plan.append("Add compost or organic matter to improve soil health")
    if not action_plan:
        action_plan.append("Monitor soil moisture every 2-3 days")
        action_plan.append("Apply balanced fertilizer based on soil test")
    
    # Rejection reasons for lower-ranked crops
    rejection_reasons = []
    if len(recommendations) > 6:
        rejected_crops = recommendations[6:]  # Take crops beyond top 6
        for crop_rec in rejected_crops[:3]:  # Only show top 3 rejections
            crop_name = crop_rec.get('crop', 'Unknown')
            reasons = []
            
            # Analyze penalties to find rejection reasons
            if 'penalties' in crop_rec:
                for penalty in crop_rec.get('penalties', [])[:2]:  # Top 2 penalties
                    if isinstance(penalty, dict):
                        reasons.append(penalty.get('reason', 'Unsuitable conditions'))
            
            if reasons:
                rejection_reasons.append({
                    "crop": crop_name,
                    "reason": reasons[0],  # Primary reason
                    "final_score": crop_rec.get('final_score', 0)
                })
    
    # (Previous) Take top recommendations by initial sort - we'll later re-sort by combined score
    
    # If no recommendations found
    if not recommendations:
        print("⚠️ No recommendations found, adding fallback recommendation")
        recommendations = [{
            "crop": "Default Recommendation",
            "season": ["Any"],
            "final_score": 50,
            "soil_health_impact": "Fair Match",
            "temp_range": "20-35",
            "rainfall_range": "500-1000",
            "soil_types": ["Any"],
            "rotation_benefit": "-",
            "matches": {},
            "penalties": [{"reason": "Fallback recommendation", "penalty": -50}]
        }]
    
    # Attach an explanation string to each recommendation using the standard
    # `build_crop_explanation` helper. This ensures every returned rec has an
    # `explanation` field; the API layer will expose only the top crop
    # explanation as `top_crop_explanation`.
    try:
        if recommendations and isinstance(recommendations, list):
            for rec in recommendations:
                try:
                    rec["explanation"] = build_crop_explanation(rec, soil_data, weather, past_crop)
                except Exception:
                    rec["explanation"] = ""
    except Exception:
        # no-op on failure
        pass

    recommendations = recommendations[:6]

    # Print debug scores for the final top recommendations
    print("🔥 FINAL SCORES FOR TOP RECOMMENDATIONS:")
    try:
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            crop_name = rec.get('crop') or rec.get('Crop') or 'Unknown'
            rule_score_val = rec.get('rule_score')
            ml_score_val = rec.get('ml_score')
            final_score_val = rec.get('final_score')
            
            print(f"[TOP] Crop: {crop_name}")
            print(f"  Rule Score: {rule_score_val}")
            print(f"  ML Score: {ml_score_val}")
            print(f"  Final Score: {final_score_val}")
    except Exception:
        pass

    # Return structured response with recommendations, action plan, and rejection reasons
    return {
        "recommendations": recommendations,
        "action_plan": action_plan,
        "rejection_reasons": rejection_reasons
    }

# ---------------- Flask Routes -----------------
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Flask server is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
        <head><title>Agricultural Advisory API</title></head>
        <body>
            <h1>🌾 Agricultural Advisory API</h1>
            <p><strong>Server Status:</strong> ✅ Running</p>
            <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
            <p><strong>API Endpoint:</strong> POST /recommend</p>
            <p><strong>Error:</strong> Template not found - {str(e)}</p>
            <h2>Quick Test:</h2>
            <button onclick="fetch('/health').then(r=>r.json()).then(d=>alert(JSON.stringify(d)))">Test Health</button>
        </body>
        </html>
        """

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/diagnostic')
def diagnostic():
    return render_template('diagnostic.html')

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """
    Google Translate API endpoint using googletrans
    
    Expected JSON:
    {
        "text": "Text to translate",
        "source_language": "en",
        "target_language": "hi"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request"
            }), 400
        
        text = data.get('text', '')
        source_language = data.get('source_language', 'en')
        target_language = data.get('target_language', 'hi')
        
        # Validate languages
        if target_language not in SUPPORTED_LANGUAGES:
            return jsonify({
                "error": f"Unsupported target language: {target_language}",
                "supported_languages": SUPPORTED_LANGUAGES
            }), 400
        
        if not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Check if translation is available
        if not TRANSLATION_AVAILABLE or translator is None:
            return jsonify({
                "error": "Google Translate service is not available",
                "message": "Translation service is temporarily unavailable"
            }), 503
        
        # Perform translation
        try:
            result = translator.translate(text, src_language=source_language, dest_language=target_language)
            
            translated_text = result['text'] if isinstance(result, dict) else result
            
            return jsonify({
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "target_language_name": SUPPORTED_LANGUAGES.get(target_language, target_language),
                "success": True
            })
        
        except Exception as translate_error:
            print(f"❌ Translation error: {translate_error}")
            return jsonify({
                "error": f"Translation failed: {str(translate_error)}",
                "success": False
            }), 500
    
    except Exception as e:
        print(f"❌ Error in translate endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500

@app.route('/api/supported-languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported translation languages"""
    return jsonify({
        "supported_languages": SUPPORTED_LANGUAGES,
        "translation_available": TRANSLATION_AVAILABLE
    })

@app.route('/api/crop-advisory', methods=['POST'])
def crop_advisory():
    """Crop advisory endpoint for frontend compatibility"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        
        # Extract location from request
        location = data.get('location', {})
        lat = location.get('latitude') or location.get('lat')
        lon = location.get('longitude') or location.get('lon')
        
        # Extract additional info
        additional_info = data.get('additionalInfo', {})
        past_crop = additional_info.get('previousCrop', 'rice')
        
        # Use current date
        date = datetime.now().strftime('%Y-%m-%d')
        
        # Validate inputs
        if lat is None or lon is None:
            return jsonify({
                "error": "Missing latitude or longitude",
                "soil_analysis": get_fallback_soil(28.6, 77.2),
                "crop_recommendations": [{
                    "crop": "Default Wheat", 
                    "season": "Rabi", 
                    "soil_match": "Fallback", 
                    "weather_match": False, 
                    "penalty": 0, 
                    "final_score": 50
                }],
                "farming_tips": ["Check soil moisture regularly"],
                "advisory_id": f"fallback_{int(datetime.now().timestamp())}",
                "confidence_score": 0.5,
                "isDemoMode": True
            }), 200
        
        print(f"🌍 Processing request for lat={lat}, lon={lon}, past_crop={past_crop}")
        
        # Get soil data with guaranteed fallback
        try:
            soil_data = get_soil_data(lat, lon)
        except Exception as e:
            print(f"❌ Soil data error: {e}")
            print("🎯 Using default fallback soil data")
            soil_data = get_fallback_soil(lat, lon)
        
        # Ensure soil_data structure
        if not soil_data or not isinstance(soil_data, dict):
            print("🎯 Using default fallback soil data")
            soil_data = get_fallback_soil(lat, lon)
        if 'soil_type' not in soil_data:
            soil_data['soil_type'] = "Unknown"
        if 'sources' not in soil_data or not isinstance(soil_data['sources'], list):
            soil_data['sources'] = ["Fallback"]
        # Ensure sources is never empty
        if len(soil_data['sources']) == 0:
            soil_data['sources'] = ["Default"]
        
        # Log final soil data
        print(f"✅ Final soil data: {soil_data}")
        
        # Get weather data
        try:
            weather_data = get_weather(lat, lon, date)
        except Exception as e:
            print(f"❌ Weather error: {e}")
            weather_data = get_fallback_weather(lat, lon, date)
        
        if not weather_data or not isinstance(weather_data, dict):
            weather_data = get_fallback_weather(lat, lon, date)
        if 'sources' not in weather_data or not isinstance(weather_data['sources'], list):
            weather_data['sources'] = ["Fallback"]
        # Ensure sources is never empty
        if len(weather_data['sources']) == 0:
            weather_data['sources'] = ["Default"]
        
        # Log final weather data
        print(f"✅ Final weather data: {weather_data}")
        
        # Get recommendations
        try:
            # Step 1: Rule-based results (for explanation)
            rule_based_result = recommend_crop_full(
                soil_data,
                past_crop=past_crop,
                weather=weather_data,
                show_all=True
            )
            
            # Extract recommendations from the new dict structure
            legacy_recommendations = rule_based_result.get("recommendations", []) if isinstance(rule_based_result, dict) else rule_based_result

            # Step 2: ML scores
            ml_scores = predict_all_ml_scores(
                soil_data,
                weather_data.get("rainfall"),
                weather_data.get("temp"),
                weather_data.get("season")
            )

            # Step 3: Environment
            user_env = {
                "temp": weather_data.get("temp"),
                "rainfall": weather_data.get("rainfall"),
                "season": weather_data.get("season"),
                "Soil_Type": soil_data.get("soil_type"),
                "pH": soil_data.get("ph"),
                "organic_matter": soil_data.get("organic_matter_pct"),
                "reliability": weather_data.get("reliability", "medium")
            }

            # Step 4: REAL HYBRID ENGINE
            hybrid_results = recommend_top_k(
                user_env=user_env,
                crops_df=pd.DataFrame(crop_data),
                last_crop=past_crop,
                k=6,
                ml_scores=ml_scores,
                rule_engine=score_crop
            )
            if hybrid_results:
                for rec in hybrid_results:
                    crop_name = rec.get("crop")
                    crop_info = next((c for c in crop_data if c.get("Crop") == crop_name), None)
                    rec["score"] = round(rec.get("final_score", 0), 1)
                    rec["temp"] = weather_data.get("temp")
                    rec["rain"] = weather_data.get("rainfall")
                    rec["season"] = weather_data.get("season")
                    if crop_info:
                        temp_value = crop_info.get("Temp")
                        if temp_value is None and crop_info.get("Temp_Min") is not None and crop_info.get("Temp_Max") is not None:
                            temp_value = f"{crop_info.get('Temp_Min')}-{crop_info.get('Temp_Max')}"
                        rain_value = crop_info.get("Rain")
                        if rain_value is None and crop_info.get("Rain_Min") is not None and crop_info.get("Rain_Max") is not None:
                            rain_value = f"{crop_info.get('Rain_Min')}-{crop_info.get('Rain_Max')}"
                        rec["temp_range"] = f"{temp_value or '—'}°C"
                        rec["rain_range"] = f"{rain_value or '—'} mm"
                        rec["rainfall_range"] = rec["rain_range"]
                    insights = generate_crop_insights(
                        rec,
                        crop_info or {},
                        weather_data,
                        soil_data
                    )
                    rec["explanation"] = insights["explanation"]
                    rec["action_plan"] = insights["action_plan"]
                    rec["risks"] = insights["risks"]
            print("🔥 HYBRID RESULTS:", hybrid_results[:2] if hybrid_results else hybrid_results)
            if hybrid_results:
                print("\U0001F525 RESPONSE:", hybrid_results[0])
            print("WEATHER DATA:", weather_data)

            if not hybrid_results:
                print("WARNING: Empty results from engine")
                recommendations = []
            else:
                recommendations = hybrid_results[:6]

            if recommendations:
                print("TOP RESULT:", recommendations[0])
        except Exception as e:
            print(f"❌ Recommendations error: {e}")
            recommendations = [{
                "crop": "Error Recovery", 
                "season": "Any", 
                "soil_match": "Error", 
                "weather_match": False, 
                "penalty": 0, 
                "final_score": 25
            }]

        # ==============================
        # 🔹 ADD ACTION PLAN TO TOP RECOMMENDATION
        # ==============================
        # Generate action plan based on conditions
        action_plan = []
        if weather_data and weather_data.get('rainfall') and weather_data.get('rainfall') < 400:
            action_plan.append("Irrigation planning required - rainfall below optimal (< 400mm)")
        if weather_data and weather_data.get('temp_anomalies'):
            action_plan.append("Temperature fluctuations detected - monitor crop closely")
        if weather_data and weather_data.get('rain_anomalies'):
            action_plan.append("Unusual rainfall patterns - ensure drainage backup ready")
        if soil_data and soil_data.get('ph') and (soil_data.get('ph') < 6.0 or soil_data.get('ph') > 7.5):
            action_plan.append(f"Soil pH adjustment needed (current: {soil_data.get('ph')})")
        if soil_data and soil_data.get('organic_matter_pct') and soil_data.get('organic_matter_pct') < 2.0:
            action_plan.append("Add compost or organic matter to improve soil health")
        if not action_plan:
            action_plan = [
                "Monitor soil moisture every 2-3 days",
                "Apply balanced fertilizer based on soil test",
                "Watch for pest symptoms and take preventive action"
            ]
        
        # Add action plan to top recommendation
        if recommendations and len(recommendations) > 0:
            recommendations[0]["action_plan"] = action_plan

        # Generate a single top-crop explanation by using the explanation attached
        # to the top recommendation (recommendations themselves should have
        # been assigned `explanation` by `recommend_crop_full`). This replaces
        # the previous bespoke logic that built top-2 and per-top-rec reasons.
        top_crop_explanation = ""
        try:
            if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                # Ensure the top recommendation has an explanation field
                top_crop_explanation = recommendations[0].get('explanation', "") or ""

                # Remove explanation fields from non-top recommendations so the
                # frontend receives the explanation only for the top crop.
                for r in recommendations[1:]:
                    if isinstance(r, dict) and 'explanation' in r:
                        try:
                            del r['explanation']
                        except Exception:
                            pass
            else:
                top_crop_explanation = ""
        except Exception:
            top_crop_explanation = ""

        # Determine rotation insights string to expose in the response
        try:
            if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                rotation_insights_summary = recommendations[0].get('rotation_insights', "") or ""
            else:
                rotation_insights_summary = (
                    f"Previous crop: {past_crop or 'Unknown'}\n"
                    "Problems caused by past crop: None recorded\n"
                    "How recommended crop helps: No specific rotation benefit recorded\n"
                    "Remaining risks: None recorded"
                )
        except Exception:
            rotation_insights_summary = (
                f"Previous crop: {past_crop or 'Unknown'}\n"
                "Problems caused by past crop: None recorded\n"
                "How recommended crop helps: No specific rotation benefit recorded\n"
                "Remaining risks: None recorded"
            )

        # Format response for frontend compatibility (requested JSON shape)
        response_data = {
            "soil_analysis": {
                "soil_type": soil_data.get('soil_type', 'Unknown'),
                "composition": soil_data.get('composition', {"clay": 30, "sand": 40, "silt": 30}),
                "sources": soil_data.get('sources', ["API"]),
                "ph_level": "6.5",  # Default values for missing analysis
                "organic_matter": "3.2",
                "nitrogen_level": "medium",
                "phosphorus_level": "medium", 
                "potassium_level": "medium",
                "moisture_content": "20.0",
                "drainage": "good",
                "confidence": 0.85,
                "impacts": {
                    "fertility": {
                        "current": "medium",
                        "trend": "stable",
                        "risks": ["nutrient depletion", "pH imbalance"],
                        "recommendations": ["regular soil testing", "balanced fertilization"]
                    },
                    "structure": {
                        "current": "good",
                        "trend": "stable",
                        "risks": ["compaction", "erosion"],
                        "recommendations": ["minimum tillage", "cover cropping"]
                    },
                    "biology": {
                        "current": "moderate",
                        "trend": "improving",
                        "risks": ["pest buildup", "beneficial organism decline"],
                        "recommendations": ["crop rotation", "organic matter addition"]
                    }
                }
            },
            "crop_recommendations": recommendations,
            "rotation_insights": rotation_insights_summary,
            "top_crop_explanation": top_crop_explanation,
            "farming_tips": [
                f"Based on {soil_data.get('soil_type', 'your soil')} soil, consider proper drainage",
                f"Current weather shows {weather_data.get('temp', 25)}°C temperature",
                f"Rainfall intensity is {weather_data.get('rainfall', 0)}mm - " + (
                    "consider supplemental irrigation" if weather_data.get('rainfall', 0) < 50 else
                    "ensure proper drainage" if weather_data.get('rainfall', 0) > 150 else
                    "maintain soil moisture"
                ),
                # Soil health recommendations
                "Implement crop rotation to prevent nutrient depletion",
                "Add organic matter to improve soil structure",
                "Monitor soil pH and adjust if necessary",
                # Disease management
                "Watch for early signs of plant diseases",
                "Maintain field hygiene to prevent disease spread",
                # Physical management
                "Minimize soil compaction during wet conditions",
                "Practice contour farming to prevent erosion",
                # Nutrient management
                "Follow balanced fertilization practices",
                "Consider soil test-based nutrient management"
            ],
            "weather_data": weather_data
        }
        
        print(f"✅ Crop Advisory API: Sending response with {len(recommendations)} recommendations")
        return {
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"❌ Crop Advisory API error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "soil_analysis": get_fallback_soil(28.6, 77.2),
            "crop_recommendations": [{
                "crop": "Emergency Fallback", 
                "season": "Any", 
                "soil_match": "Error", 
                "weather_match": False, 
                "penalty": 0, 
                "final_score": 1
            }],
            "farming_tips": ["Server error occurred, please try again"],
            "advisory_id": f"error_{int(datetime.now().timestamp())}",
            "confidence_score": 0.1,
            "isDemoMode": True
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json or {}

        def _try_float(value):
            try:
                if value is None:
                    return None
                if isinstance(value, str) and not value.strip():
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        lat = _try_float(data.get('lat'))
        lon = _try_float(data.get('lon'))
        past_crop = data.get('past_crop')
        date = data.get('date')

        if lat is None or lon is None:
            city = data.get('city') or data.get('City')
            state = data.get('state') or data.get('State')
            resolved = geocode_location(city, state)
            if resolved:
                lat, lon = resolved
                print(f"📍 Manual location resolved to lat={lat}, lon={lon}")
            else:
                print("⚠️ Manual location missing/failed; falling back to Delhi coordinates")
                lat, lon = 28.6, 77.2

        print(f"🌍 Processing request for lat={lat}, lon={lon}, past_crop={past_crop}")

        # Get soil data with guaranteed fallback
        try:
            soil_data = get_soil_data(lat, lon)
        except Exception as e:
            print(f"❌ Soil data error: {e}")
            print("🎯 Using default fallback soil data")
            soil_data = get_fallback_soil(lat, lon)
        
        # Ensure soil_data always has the required structure
        if not soil_data or not isinstance(soil_data, dict) or 'soil_type' not in soil_data:
            print("⚠️ Invalid soil data, using default fallback")
            soil_data = get_fallback_soil(lat, lon)
        
        # Ensure composition exists
        if 'composition' not in soil_data or not soil_data['composition']:
            soil_data['composition'] = {"clay": 30.0, "sand": 40.0, "silt": 30.0}
        
        try:
            soil_health_summary = calculate_soil_health_score(soil_data, {})
            soil_data['soil_health'] = {
                "score": soil_health_summary.get('score'),
                "status": soil_health_summary.get('risk_level'),
                "risk_level": soil_health_summary.get('risk_level'),
                "recommendation": soil_health_summary.get('recommendation'),
                "impacts": soil_health_summary.get('impacts', [])
            }
        except Exception as e:
            print(f"⚠️ Soil health scoring failed: {e}")

        print(f"✅ Final soil data: {soil_data}")

        # Get weather data with guaranteed fallback
        try:
            weather_data = get_weather(lat, lon, date)
        except Exception as e:
            print(f"❌ Weather data error: {e}")
            print("🎯 Using fallback weather data")
            weather_data = get_fallback_weather(lat, lon, date)
        
        # Ensure weather_data always has the required structure  
        if not weather_data or not isinstance(weather_data, dict):
            print("⚠️ Invalid weather data, using default fallback")
            weather_data = get_fallback_weather(lat, lon, date)
            
        # Ensure sources is always a valid array
        if 'sources' not in weather_data or not isinstance(weather_data['sources'], list):
            weather_data['sources'] = ["Fallback"]
        
        print(f"✅ Final weather data: {weather_data}")

        # ==============================
        # 🔹 SIMULATION OVERRIDE
        # ==============================
        simulate_temp = data.get('simulate_temp')
        simulate_rain = data.get('simulate_rain')
        if simulate_temp is not None:
            weather_data["temp"] = float(simulate_temp)
            print(f"🎛️ Simulated temp override: {simulate_temp}°C")
        if simulate_rain is not None:
            weather_data["rainfall"] = float(simulate_rain)
            print(f"🎛️ Simulated rainfall override: {simulate_rain}mm")

        # ==============================
        # 🔹 PREPARE ENVIRONMENT FOR HYBRID ENGINE
        # ==============================
        user_env = {
            "temp": weather_data.get("temp"),
            "rainfall": weather_data.get("rainfall"),
            "season": weather_data.get("season"),
            "Soil_Type": soil_data.get("soil_type"),
            "pH": soil_data.get("ph"),
            "organic_matter": soil_data.get("organic_matter_pct"),
            "reliability": weather_data.get("reliability", "medium"),
            "temp_anomalies": weather_data.get("temp_anomalies", False),
            "rain_anomalies": weather_data.get("rain_anomalies", False)
        }

        # ==============================
        # 🔹 CALCULATE ML SCORES
        # ==============================
        ml_scores = predict_all_ml_scores(
            soil_data,
            weather_data.get("rainfall"),
            weather_data.get("temp"),
            weather_data.get("season")
        )

        # ==============================
        # 🔹 CALL HYBRID RECOMMENDATION ENGINE
        # ==============================
        # This orchestrates everything: rule scoring, ML scoring, soil scoring, rotation scoring
        hybrid_results = recommend_top_k(
            user_env=user_env,
            crops_df=pd.DataFrame(crop_data),
            last_crop=past_crop,
            k=6,
            ml_scores=ml_scores,
            rule_engine=score_crop,
            rotation_engine=get_rotation_score
        )
        if hybrid_results:
            for rec in hybrid_results:
                crop_name = rec.get("crop")
                crop_info = next((c for c in crop_data if c.get("Crop") == crop_name), None)
                rec["score"] = round(rec.get("final_score", 0), 1)
                rec["temp"] = weather_data.get("temp")
                rec["rain"] = weather_data.get("rainfall")
                rec["season"] = weather_data.get("season")
                if crop_info:
                    temp_value = crop_info.get("Temp")
                    if temp_value is None and crop_info.get("Temp_Min") is not None and crop_info.get("Temp_Max") is not None:
                        temp_value = f"{crop_info.get('Temp_Min')}-{crop_info.get('Temp_Max')}"
                    rain_value = crop_info.get("Rain")
                    if rain_value is None and crop_info.get("Rain_Min") is not None and crop_info.get("Rain_Max") is not None:
                        rain_value = f"{crop_info.get('Rain_Min')}-{crop_info.get('Rain_Max')}"
                    rec["temp_range"] = f"{temp_value or '—'}°C"
                    rec["rain_range"] = f"{rain_value or '—'} mm"
                    rec["rainfall_range"] = rec["rain_range"]
                insights = generate_crop_insights(
                    rec,
                    crop_info or {},
                    weather_data,
                    soil_data
                )
                rec["explanation"] = insights["explanation"]
                rec["action_plan"] = insights["action_plan"]
                rec["risks"] = insights["risks"]
        print("🔥 HYBRID RESULTS:", hybrid_results[:2] if hybrid_results else hybrid_results)
        if hybrid_results:
            print("\U0001F525 RESPONSE:", hybrid_results[0])
        print("WEATHER DATA:", weather_data)

        if not hybrid_results:
            print("WARNING: Empty results from engine")
            recommendations = []
        else:
            recommendations = hybrid_results[:6]

        if recommendations:
            print("TOP RESULT:", recommendations[0])

        # ==============================
        # 🔹 ADD EXPLANATIONS TO TOP RECOMMENDATION
        # ==============================
        try:
            if recommendations and isinstance(recommendations, list):
                for i, rec in enumerate(recommendations):
                    if i == 0:
                        # Top crop gets explanation
                        if not rec.get("explanation"):
                            rec["explanation"] = build_crop_explanation(rec, soil_data, weather_data, past_crop)
                    else:
                        # Other crops don't get explanation (too verbose)
                        rec.pop("explanation", None)
        except Exception as e:
            print(f"⚠️ Failed to add explanations: {e}")

        # Ensure only the top recommendation carries the explanation field
        top_crop_explanation = ""
        try:
            if recommendations and isinstance(recommendations, list):
                top_crop_explanation = recommendations[0].get('explanation', "") or ""
                for rec in recommendations[1:]:
                    if isinstance(rec, dict) and 'explanation' in rec:
                        rec.pop('explanation', None)
        except Exception:
            top_crop_explanation = ""

        try:
            if recommendations and isinstance(recommendations, list):
                rotation_insights_summary = recommendations[0].get('rotation_insights', "") or ""
            else:
                rotation_insights_summary = (
                    f"Previous crop: {past_crop or 'Unknown'}\n"
                    "Problems caused by past crop: None noted\n"
                    "How the recommended crop helps recover: Supports soil recovery and breaks pest cycles\n"
                    "Remaining risks for this new crop: None noted"
                )
        except Exception:
            rotation_insights_summary = (
                f"Previous crop: {past_crop or 'Unknown'}\n"
                "Problems caused by past crop: None noted\n"
                "How the recommended crop helps recover: Supports soil recovery and breaks pest cycles\n"
                "Remaining risks for this new crop: None noted"
            )

        weather = {
            "temp": weather_data.get("temp"),
            "rainfall": weather_data.get("rainfall"),
            "season": weather_data.get("season")
        }
        soil = {
            "soil_type": soil_data.get("soil_type"),
            "ph": soil_data.get("ph"),
            "organic_matter": soil_data.get("organic_matter") or soil_data.get("organic_matter_pct"),
            "composition": soil_data.get("composition"),
            "soil_health": soil_data.get("soil_health")
        }

        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            rec["score"] = round(rec.get("final_score", 0), 1)
            if weather:
                rec["temp"] = weather.get("temp")
                rec["rain"] = weather.get("rainfall")
                rec["season"] = weather.get("season")

        print("FINAL WEATHER:", weather)
        print("FINAL SOIL:", soil)
        
        print(f"✅ Sending response with {len(recommendations)} recommendations")
        return {
            "recommendations": recommendations,
            "weather": weather_data,
            "soil": soil_data,
            "weather_data": weather_data,
            "soil_data": soil_data
        }

    except Exception as e:
        print(f"❌ General error in recommend route: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a safe fallback response
        fallback_soil = get_fallback_soil(28.6, 77.2)  # Default Delhi coordinates
        fallback_weather = get_fallback_weather(28.6, 77.2)
        
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "soil": fallback_soil,
            "soil_data": fallback_soil,  # Include both keys for compatibility
            "weather": fallback_weather,
            "weather_data": fallback_weather,  # Include both keys for compatibility
            "recommendations": [{
                "crop": "Error - using default data", 
                "season": "Unknown", 
                "soil_match": "Unknown", 
                "weather_match": False, 
                "penalty": 0, 
                "final_score": 0
            }]
        }), 500

if __name__ == "__main__":
    import os
    import signal
    
    print("🚀 Starting Agricultural Advisory API Server...")
    print("✅ Server will be available at:")
    print("   - Local: http://127.0.0.1:5000")
    print("   - External: https://friendly-xylophone-9p5jvg69j6gcxgpw-5000.app.github.dev/")
    print("✅ Health check: /health")
    print("✅ API endpoint: POST /recommend")
    print("✅ CORS enabled for frontend access")
    print("✅ Fallback systems active")
    
    def signal_handler(sig, frame):
        print('\n🛑 Server shutting down gracefully...')
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # More robust Flask configuration
        app.config['ENV'] = 'development'
        app.config['TESTING'] = False
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        
        print("🌐 Starting server on 0.0.0.0:5000...")
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Disable debug to prevent reloader issues
            use_reloader=False, 
            threaded=True,
            processes=1
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

