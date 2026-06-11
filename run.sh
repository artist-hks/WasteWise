#!/bin/bash
echo "=== WasteWise Setup ==="
pip install -r requirements.txt -q
echo "Generating synthetic data..."
python data/simulate_data.py
echo "Training forecasting models..."
python models/waste_forecaster.py
echo "Launching dashboard..."
python -m streamlit run app/main.py --server.port 8501
