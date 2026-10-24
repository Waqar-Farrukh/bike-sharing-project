# Bike Sharing Rental Prediction – Assignment 3

Regression task: predict the number of bikes rented in one hour (`cnt`) from weather and calendar data.
Five algorithms compared with 10-fold cross validation and RMSE on the full dataset (17,379 hours).

Algorithms (none used in the Car Price notebook): SVR, Ridge Regression, Extra Trees, AdaBoost, HistGradientBoosting.

## Files

| File | Purpose |
|---|---|
| `bike_sharing_10fold_cv.ipynb` | Main notebook (already executed, ready for Google Colab) |
| `hour.csv` | Dataset (UCI Bike Sharing, hourly file) |
| `bike_rental_model.pkl` | Trained final model (best algorithm, trained on all data) |
| `model_info.json` | Best algorithm, cross-validation RMSE, scikit-learn version |
| `app.py` | Streamlit web app |
| `bike_model.py` | Shared pipeline and algorithm definitions used by the app |
| `requirements.txt` | Libraries for the web app |

## Run the web app on your computer

```
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or newer is required.

If the saved model was made with a different scikit-learn version, the app rebuilds it from `hour.csv` automatically in a few seconds.

## If the app shows "the model was rebuilt"

This is not an error. It means your scikit-learn version differs from the one that saved the model, so the app trained the same model again from `hour.csv`. To make the app load the saved model directly, run this once inside the project folder, using the same Python environment you use for Streamlit:

```
python -c "from bike_model import *; save_model(train_final(read_info()['best_algorithm'])); write_info(read_info())"
```

Then check your version with `pip show scikit-learn` and set the same version in `requirements.txt`, for example `scikit-learn==1.7.2`, before uploading to GitHub.

## Run the notebook on Google Colab

1. Open https://colab.research.google.com and choose File, Upload notebook.
2. Select `bike_sharing_10fold_cv.ipynb`, then Runtime, Run all.
3. When asked, upload `hour.csv`.
4. In Step 10.3 set `DOWNLOAD_MODEL = True` and run that cell to download your own `bike_rental_model.pkl` and `model_info.json`.

## Deploy to the cloud (Streamlit Community Cloud, free)

1. Create a public GitHub repository and upload every file in this folder (the notebook is optional for deployment).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click Create app, pick the repository and branch, set the main file path to `app.py`.
4. Under Advanced settings choose Python 3.12, then Deploy.
5. Copy the live link into the first line of the notebook.

## Submission (Google Classroom)

Upload the notebook (with the live link on its first line), `hour.csv` and `bike_rental_model.pkl`.

## Data source

Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013), pp. 1-15, Springer. Hosted by the UCI Machine Learning Repository.
