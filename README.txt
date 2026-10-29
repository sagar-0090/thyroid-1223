THYRO-CVD MULTI-PAGE HTML/CSS/JS + FLASK STARTER
=================================================

Pages:
- /              Overview dashboard
- /assessment    Multi-section assessment form and results
- /research      Dataset/model metrics and workflow

Files:
- server.py
- templates/index.html
- templates/assessment.html
- templates/research.html
- static/style.css
- static/app.js
- static/assessment.js
- static/research.js
- requirements.txt

Install/run:
1. Copy these files/folders into C:\ty\Thyroid_Project.
2. Keep existing data/, output/, model_artifacts/, and scripts.
3. In PowerShell:
   cd "C:\ty\Thyroid_Project"
   .\.venv312\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv312\Scripts\python.exe server.py
4. Open http://127.0.0.1:5000

If .venv312 is not the environment with working imports, use the working environment's Python executable instead.

CATEGORY LABEL WARNING:
The website does not display raw numeric category codes as human-readable meaning. Gender and binary yes/no fields are normalized for common encodings. For Race and Thyroid_Diagnosis, the backend does not guess the meaning of numeric codes. If your source CSV stores those categories as 1.0, 2.0, etc., consult the original dataset's data dictionary and add the correct mapping in clean_category() before using the website. Do not invent a mapping. For production, store explicit human-readable category labels in source data or configure verified mappings.

MODEL:
The backend retrains a Random Forest on startup from the largest configured candidate CSV containing Cardiovascular_Disease. Confirm that the chosen dataset contains all required columns and is the intended source. This is a starter integration and does not guarantee identical predictions to the original Streamlit app. The model and rule-based thresholds are for research demonstration only, not medical diagnosis.
