# Dataset

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset. It contains 101,766 hospital encounters from 130 US hospitals; the project data has 71,518 unique patients. One patient can have several encounters.

The original files are kept unchanged in `raw/`:

- `diabetic_data.csv` — encounter records used for the analysis.
- `IDS_mapping.csv` — descriptions of coded admission and discharge IDs.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cleaning and feature preparation are done in code, not by editing the CSV files.

To download the data, use the **Download** button on the UCI page or the [direct ZIP download](https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip). Extract both CSV files into `data/raw/`.
