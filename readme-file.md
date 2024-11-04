# Utility Risk Prediction Dashboard

## Project Overview
This dashboard helps utility companies predict and mitigate operational risks by analyzing weather patterns and historical incident data. It uses machine learning to identify high-risk periods and provides actionable insights for risk mitigation.

## Features
- Real-time weather data integration
- Risk prediction using Random Forest classification
- Interactive visualizations of risk levels
- Correlation analysis of risk factors
- High-risk period alerts
- Historical data analysis

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/utility-risk-dashboard.git
cd utility-risk-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenWeatherMap API key:
- Sign up at OpenWeatherMap to get an API key
- Replace `YOUR_API_KEY` in `main.py` with your actual API key

## Usage
1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. The dashboard will automatically:
   - Fetch current weather data
   - Load historical data
   - Train the risk prediction model
   - Display visualizations and alerts

## Data Sources
- Weather data: OpenWeatherMap API
- Historical weather data: NOAA database
- Utility incident data: Simulated based on historical patterns

## Project Structure
```
utility-risk-dashboard/
├── main.py                 # Main application file
├── requirements.txt        # Package dependencies
├── README.md              # Project documentation
├── data/
│   ├── historical_weather.csv
│   └── utility_incidents.csv
├── models/
│   ├── risk_model.joblib
│   └── scaler.joblib
└── notebooks/
    └── data_analysis.ipynb
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenWeatherMap for providing weather data API
- Streamlit for the dashboard framework
- scikit-learn for machine learning capabilities
