import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class UtilityRiskDashboard:
    def __init__(self):
        self.weather_data = None
        self.incident_data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_weather_data(self, city="New York"):
        """Fetch real weather data from OpenWeatherMap API"""
        API_KEY = "YOUR_API_KEY"  # Replace with your OpenWeatherMap API key
        base_url = "http://api.openweathermap.org/data/2.5/forecast"
        
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_list = []
            
            for item in data['list']:
                weather_list.append({
                    'datetime': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind']['speed'],
                    'weather_condition': item['weather'][0]['main']
                })
            
            self.weather_data = pd.DataFrame(weather_list)
            return True
        return False
    
    def load_historical_data(self):
        """Load and combine historical weather and incident data"""
        # Load historical weather data (you would typically load this from a CSV)
        historical_weather = pd.read_csv('data/historical_weather.csv')
        
        # Load incident data (you would typically load this from a CSV)
        incident_data = pd.read_csv('data/utility_incidents.csv')
        
        # Combine the datasets
        self.incident_data = pd.merge(
            historical_weather, 
            incident_data,
            on='datetime',
            how='left'
        )
        
    def train_model(self):
        """Train the risk prediction model"""
        features = ['temperature', 'humidity', 'wind_speed']
        X = self.incident_data[features]
        y = self.incident_data['incident_occurred']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Save the model
        joblib.dump(self.model, 'models/risk_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
    def predict_risk(self, weather_data):
        """Predict risk levels based on weather conditions"""
        features = ['temperature', 'humidity', 'wind_speed']
        X = weather_data[features]
        X_scaled = self.scaler.transform(X)
        
        risk_probabilities = self.model.predict_proba(X_scaled)
        return risk_probabilities[:, 1]  # Return probability of incident occurring

def main():
    st.title("Utility Risk Prediction Dashboard")
    
    dashboard = UtilityRiskDashboard()
    
    # Fetch current weather data
    if dashboard.fetch_weather_data():
        st.success("Successfully fetched weather data")
        
        # Load historical data and train model
        dashboard.load_historical_data()
        dashboard.train_model()
        
        # Make predictions
        risk_levels = dashboard.predict_risk(dashboard.weather_data)
        dashboard.weather_data['risk_level'] = risk_levels
        
        # Create visualizations
        st.subheader("Risk Level Forecast")
        fig = px.line(dashboard.weather_data, 
                     x='datetime', 
                     y='risk_level',
                     title='Predicted Risk Levels Over Time')
        st.plotly_chart(fig)
        
        # Create risk heatmap
        st.subheader("Risk Factors Heatmap")
        correlation_matrix = dashboard.incident_data[
            ['temperature', 'humidity', 'wind_speed', 'incident_occurred']
        ].corr()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu'
        ))
        st.plotly_chart(fig_heatmap)
        
        # High-risk alerts
        st.subheader("High-Risk Alerts")
        high_risk_periods = dashboard.weather_data[
            dashboard.weather_data['risk_level'] > 0.7
        ]
        
        if not high_risk_periods.empty:
            st.warning("High-risk periods detected!")
            st.write(high_risk_periods[['datetime', 'risk_level', 'weather_condition']])
        else:
            st.info("No high-risk periods detected in the forecast.")

if __name__ == "__main__":
    main()
