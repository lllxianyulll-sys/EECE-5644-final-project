# EECE5644 Final Project – Chicago Bike Demand Prediction

This repository contains the source code for my EECE5644 Final Project.  
The goal of the project is to build multi-output regression models to
predict hourly bike-sharing demand across Chicago community areas.

## Data Sources
- **Divvy Trip Data (December 2023)**  
  Provided by Lyft/Divvy Open Data Portal.  
  URL: https://divvy-tripdata.s3.amazonaws.com/202312-divvy-tripdata.zip

- **Chicago Weather Data**  
  Source: Chicago Weather Database (Kaggle).  
  URL: https://www.kaggle.com/datasets/curiel/chicago-weather-database

- **Chicago Community Area Boundaries (GeoJSON)**  
  Provided by City of Chicago Data Portal.  
  URL: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-Map/cauq-8yn6


File Structure
EECE5644_Project/
│
├── EECE5644_Project.py        # Main script with full data pipeline and model training
├── divvy_trip_data_202312.csv # Trip dataset (must be downloaded separately)
├── data.csv                   # Weather dataset
├── Boundaries_Community.geojson # Chicago community area boundaries

How to Run
Make sure the dataset paths in the script are correct.

use absolute paths 

For example in the .py file
# ================== File paths ==================
TRIP_CSV_PATH = r"C:\Users\ALIENWARE\Desktop\divvy_trip_data_202312.csv"
WEATHER_CSV_PATH = r"C:\Users\ALIENWARE\Desktop\data.csv"
COMMUNITY_GEO_PATH = r"C:\Users\ALIENWARE\Desktop\Boundaries_-_Community_Areas_20251130.geojson"

Run the project:

python EECE5644_Project.py

This script will:
	Load and clean Divvy trip data
	Merge weather and geospatial information
	Aggregate hourly demand
	Train multiple regression models (Ridge, Decision Tree, Random Forest, XGBoost, Gradient Boosting)
	Evaluate performance using R² and RMSE
	Print summary results to the console



















