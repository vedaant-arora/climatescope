from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

DATASET = "nelgiriyewithana/global-weather-repository"
DOWNLOAD_PATH = "data"

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    DATASET,
    path=DOWNLOAD_PATH,
    unzip=True,
    force=True
)

print("Dataset updated successfully.")
