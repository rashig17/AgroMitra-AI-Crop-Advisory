# AgroMitra – AI-Powered Crop Advisory Platform


[![Flask](https://img.shields.io/badge/Flask-API-green)](https://flask.palletsprojects.com/) [![Python](https://img.shields.io/badge/Python-ML-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



## One-line project description
An AI-powered, multilingual crop recommendation platform that combines weather + soil intelligence with a hybrid recommendation engine and a trained ML model.

AgroMitra is an AI-powered decision support platform designed to help farmers make informed crop planning decisions by combining weather data, soil analysis, machine learning, and rule-based recommendation systems.


## Overview
AgroMitra-AI helps farmers and agronomists generate crop recommendations based on:
- weather data retrieved through API integrations with intelligent fallback mechanisms,
- **soil** properties (texture, pH, organic matter) from external services,
- **crop ranking** using a **rule-based + ML hybrid** recommendation pipeline,
- **crop rotation** insights using past crop/counter information.

The application is implemented as a **Flask web API** with a web interface for generating AI-powered crop recommendations.


## Why I Built This
Small farmers often make crop decisions with limited local context. This project automates evidence-driven advisory by orchestrating multiple data sources and scoring strategies.

## Problem Statement
Crop suitability depends on interacting factors such as climate (temperature/rainfall), soil properties, and rotation/management constraints. Many existing tools are either:
- not multilingual,
- too complex to use,
- or rely on a single data source or a single scoring method.

## Product Vision
A practical, farmer-first advisory platform that:
- delivers explainable recommendations,
- supports local languages,
- scales via API orchestration,
- continuously improves via data-driven ML.

## Key Features
- **AI-powered multilingual crop advisory**
- **Curated agricultural knowledge base developed through extensive domain research**
- **Weather API integration** (Open-Meteo + caching + fallbacks)
- **Soil API integration** (SoilGrids with REST + WMS fallback + estimation fallback)
- **Recommendation Engine** (hybrid orchestration)
- **Crop ranking** with explainable signals and rotation considerations
- **Flask web application** + CORS-ready API endpoints


## Applications
- Farmer advisory systems and demo kiosks
- Agritech hackathons / SIH portfolio projects
- Rapid prototyping for crop recommendation research
- Multilingual agronomy assistants

## AI Workflow
1. **User inputs** location + optional past crop
2. **Weather retrieval** (or fallback)
3. **Soil retrieval** (texture + pH/OM, or estimation fallback)
4. **Feature engineering** for ML environment vector
5. **Hybrid scoring**: rule-based engine + ML blending + rotation scoring
6. **Crop ranking** and **explanations** for the top recommendation
7. UI renders recommended crops + insights

## System Architecture
```text
User
   │
   ▼
Flask Web Application
   │
   ├── Weather APIs
   ├── Soil APIs
   ▼
Recommendation Engine
   ▼
Machine Learning Model
   ▼
Crop Ranking
   ▼
Recommendation Dashboard
```


## API Integrations
- **Weather:** Open-Meteo (forecast/current) with smart caching and fallbacks
- **Soil:** SoilGrids (REST-first, WMS fallback)
- **Translation (optional):** Google Translate via `googletrans`

## Machine Learning Model
The recommendation engine combines machine learning with rule-based scoring to generate personalized crop recommendations.

The trained model is generated during training and loaded by the recommendation engine during inference. Large model artifacts are excluded from the repository because of GitHub file size limitations.

## Dataset

The recommendation engine is supported by a curated agricultural knowledge base developed through extensive research from publicly available agricultural resources.

The dataset includes:

- Crop characteristics
- Temperature and rainfall suitability
- Soil compatibility
- Seasonal information
- Crop impact data
- Agronomic metadata used by the recommendation engine

Unlike standard benchmark datasets, this knowledge base was manually curated, validated, and structured specifically for AgroMitra to support explainable and practical crop recommendations.

If appropriate, the knowledge base was compiled from multiple trusted agricultural references and standardized into a unified format for the recommendation engine.


## Tech Stack

- **Backend:** Flask, Flask-CORS
- **ML:** scikit-learn, xgboost, joblib
- **Data processing:** NumPy, Pandas
- **Utilities:** requests, urllib3 retry strategy
- **Frontend:** Tailwind (CDN), vanilla JS

## Project Structure
- `app.py` — Flask routes + API orchestration
- `engine/` — recommendation, rotation, and scoring engines
- `models/` — model artifacts + training reports
- `data/`
  - `crops.json` — curated agricultural knowledge base
  - `crop_impacts.json` — crop impact metadata
  - `crops_csv.csv` — structured crop dataset used by the recommendation engine
- `training/` — training scripts and experiments
- `templates/` — UI template(s)
- `reports/` — experiment outputs/figures


## Installation
```bash
git clone https://github.com/rashig17/AgroMitra-AI-Crop-Advisory-Platform.git
cd AgroMitra-AI-Crop-Advisory-Platform
pip install -r requirements.txt
python app.py
```


## Quick Start
1. Set required environment variables (for live API access):
   - `OPENWEATHERMAP_API_KEY` (if used in your configuration)
2. Run the server:
```bash
python app.py
```
3. Open the app in your browser.

## Running the Application
- Local endpoint: `http://127.0.0.1:5000/`
- Health check: `GET /health`
- Recommendation API: `POST /recommend`

## Demo & Screenshots
- **Demo Video:** *Coming Soon*
- **Home Screen:** *Coming Soon*
- **Recommendation Dashboard:** *Coming Soon*
- **Crop Results:** *Coming Soon*
- **Architecture Diagram:** *Coming Soon*

## Product Design
AgroMitra was designed around practical farmer workflows rather than only machine learning predictions.

Key product considerations include:

- Multilingual accessibility
- Explainable crop recommendations
- Weather-aware decision making
- Soil-based personalization
- Intelligent API fallback mechanisms
- Scalable API-driven architecture

## AI-Assisted Development
Generative AI tools were used throughout development to improve productivity and accelerate implementation.

- ChatGPT — debugging, feature implementation, and problem solving
- Claude — documentation and technical explanations
- GitHub Copilot — code completion and API integration
- Gemini — research and implementation ideas

## Impact
AgroMitra demonstrates how machine learning, external APIs, recommendation systems, and product thinking can be combined to build practical AI-powered decision-support tools for agriculture.

## Future Improvements

- Add more weather sources (multi-endpoint ensemble)
- Extend UI to visualize confidence and per-factor contribution
- Add model retraining automation and evaluation dashboards
- Improve soil inference with additional depth layers

## Author

Rashi Gupta

## License

Copyright © 2026 Rashi Gupta. All Rights Reserved.

This repository is made publicly available for portfolio and educational viewing purposes only. No permission is granted to copy, modify, redistribute, or use the source code, datasets, or documentation without prior written permission from the author.

