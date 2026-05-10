# Model Evaluation

## Random Forest
- R2: 0.335
- RMSE: 0.173
- MAE: 0.040

## XGBoost
- R2: 0.075
- RMSE: 0.204
- MAE: 0.046

## Feature Importance (Top 10)
| Feature          |   SHAP_Importance |
|:-----------------|------------------:|
| decimalLongitude |       0.00518473  |
| decimalLatitude  |       0.00240308  |
| month            |       8.57346e-05 |
| NDVI             |       0           |
| canopy_cover     |       0           |
| soil_moisture    |       0           |
| RH2M             |       0           |
| PRECTOTCORR      |       0           |
| WS2M             |       0           |
| T2M              |       0           |