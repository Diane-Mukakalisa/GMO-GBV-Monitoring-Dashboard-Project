# Interactive Gender-Based Violence (GBV) Monitoring Dashboard

## Project Overview
This project presents an interactive **Gender-Based Violence (GBV) Monitoring Dashboard** developed to support the Gender Monitoring Office (GMO) in generating timely, data-driven insights for GBV prevention, response, and policy advocacy. The dashboard consolidates GBV case management data into a single analytics platform, enabling trend analysis, geographic hotspot identification, and case monitoring.

The project addresses challenges related to fragmented reporting systems and limited real-time analytics by providing a user-friendly, visual, and evidence-based decision-support tool.

---

## Repository Structure
├── data/
│ ├── raw/ # Original GBV case management data (restricted / not shared publicly)
│ ├── processed/ # Cleaned and processed datasets used by the dashboard

│
├── Data storage/
│ ├── pgAdmin 4-servers-PostgreSQL 17-Databases-GBV Monitoring-Schemas-Public-Tables-gbv_case_management
│ 
│
├── app/
│ ├── gbv_advanced_dashboard.py # Streamlit application code
│
├── assets/
│ ├── images/ # Dashboard screenshots (for demo and reporting)
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies


---

## Data Description
The dataset consists of GBV case reports received by the Gender Monitoring Office (GMO) covering the period **2022–2025**.

### Key Variables
- Case date
- District
- Victim age and sex
- Type of violence
- Reporting channel
- Case status and follow-up
- Case summary

> ⚠️ Note: Due to the sensitive nature of GBV data, raw datasets are not publicly shared.

---

## Data Processing 
Data processing steps included:
- Standardization of dates, phone numbers, and location names
- Removal of corrupted or incomplete records
- Harmonization of categorical variables
- Creation of derived variables (year, month) for time-series analysis

---

## Methodology
- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Libraries:** Pandas, Plotly, Matplotlib  

Descriptive analytics techniques were used to analyze:
- Temporal trends in GBV cases
- Geographic distribution by district
- Demographic patterns (age, sex)
- Case progression (reported, referred, returned)

Design decisions focused on usability, clarity, and relevance for institutional decision-makers.

---

## Results and Outputs
The main output is an interactive GBV Monitoring Dashboard featuring:
- KPI (Key Performance Indicators) summary indicators
- GBV trends over time
- District-level bar charts
- District heatmaps highlighting GBV hotspots
- Cases by reporting channel used
- Disaggregations
- Case progress tracking

---

## Live Application
- **Local access:** http://localhost:8501  

---
## Team Members
Diane Mukakalisa – Data Cleaning, Analytics, Dashboard Development

## How to Run the Application Locally
1. Clone the repository:
```bash
git clone https://github.com/Diane-Mukakalisa/GMO-GBV-Monitoring-Dashboard-Project.git


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run gbv_advanced_dashboard.py

---

# PART 2: Short Demo Script (2 Minutes)

Short video available at: Watch Demo Video

This demonstration presents an interactive Gender-Based Violence (GBV) Monitoring Dashboard developed to support the Gender Monitoring Office in evidence-based monitoring and decision-making. The demo shows how GBV case data from 2022 to 2025 are consolidated into a single analytics platform to visualize trends over time, geographic distribution by district, hotspot identification using heatmaps, and case progress from reporting to referral and follow-up. Through key performance indicators and clear visualizations, the dashboard enables timely identification of high-risk districts, supports targeted interventions, improves coordination among stakeholders, and strengthens data-driven policy advocacy while ensuring ethical handling and confidentiality of sensitive GBV information nationwide

