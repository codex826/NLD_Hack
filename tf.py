import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transit Intelligence & TFT Forecasting", layout="wide")
st.title("🚇 Network Diagnostics & TFT Deep Learning")

# --- INITIALIZE SESSION STATE ---
if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = None
if 'historical_avg' not in st.session_state:
    st.session_state['historical_avg'] = 0

# --- TFT CORE COMPONENTS ---
class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation='sigmoid')

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation='elu')
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        x = self.glu(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        return self.layer_norm(x + inputs)

# --- DATA HELPERS ---
def get_file(uploaded_files, name_substring):
    for f in uploaded_files:
        if name_substring.lower() in f.name.lower():
            return pd.read_csv(f)
    return None

# --- SIDEBAR ---
st.sidebar.header("1. Data Input")
uploaded_files = st.sidebar.file_uploader("Upload all 5 Project CSVs", accept_multiple_files=True, type=['csv'])

if len(uploaded_files) >= 5:
    df_ridership = get_file(uploaded_files, "Ridership")
    df_traffic = get_file(uploaded_files, "Traffic")
    df_mapping = get_file(uploaded_files, "Mapping")
    df_stops = get_file(uploaded_files, "Stops")
    df_routes = get_file(uploaded_files, "Routes")

    if all(v is not None for v in [df_ridership, df_traffic, df_mapping, df_stops, df_routes]):
        # Data Processing
        df_ridership['Date'] = pd.to_datetime(df_ridership['Date'])
        df_traffic['Date'] = pd.to_datetime(df_traffic['Date'])

        df = pd.merge(df_ridership, df_traffic, on='Date', how='inner')
        df = pd.merge(df, df_mapping, on=['Route_ID', 'Stop_ID'], how='inner')
        df = pd.merge(df, df_stops, on='Stop_ID', how='left')
        df = pd.merge(df, df_routes, on='Route_ID', how='left')

        df['Net_Flow'] = df['Boarding_Count'] - df['Alighting_Count']
        target_route = st.sidebar.selectbox("Select Route ID", options=df['Route_ID'].unique())
        route_df = df[df['Route_ID'] == target_route].sort_values('Date')

        # --- PREDICTION SETTINGS ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. Prediction Query")
        query_date = st.sidebar.date_input("Query Specific Date", 
                                          value=datetime(2025, 7, 1),
                                          min_value=datetime(2025, 7, 1),
                                          max_value=datetime(2025, 12, 31))

        # --- MODEL BUILDER ---
        def build_tft_model(input_shape, units=64, dropout_rate=0.1):
            inputs = layers.Input(shape=input_shape)
            x = GatedResidualNetwork(units, dropout_rate)(inputs)
            x = layers.LSTM(units, return_sequences=True)(x)
            attn = layers.MultiHeadAttention(num_heads=4, key_dim=units)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            x = layers.Flatten()(x)
            x = GatedResidualNetwork(units, dropout_rate)(x)
            outputs = layers.Dense(1)(x)
            model = Model(inputs, outputs)
            model.compile(optimizer='adam', loss='mse')
            return model

        # --- TABS ---
        tab_diag, tab_pred = st.tabs(["📊 Network Diagnostics", "🔮 H2 2025 Predictions"])

        with tab_diag:
            st.header("Baseline Network Diagnostics")
            c1, c2 = st.columns(2)
            with c1:
                hist_daily = route_df.groupby('Date')['Boarding_Count'].sum().reset_index()
                st.plotly_chart(px.line(hist_daily, x='Date', y='Boarding_Count', title="Historical Demand Trend"), use_container_width=True)
            with c2:
                stop_imb = route_df.groupby('Stop_Name')['Net_Flow'].mean().reset_index()
                st.plotly_chart(px.bar(stop_imb, x='Stop_Name', y='Net_Flow', title="Stop-level Structural Imbalance"), use_container_width=True)

        with tab_pred:
            st.header(f"TFT Prediction Dashboard (Route {target_route})")
            
            if st.button("🚀 Run Growth-Aware Forecast"):
                with st.spinner("Training Temporal Fusion Transformer..."):
                    # Prep Data
                    daily_data = route_df.groupby('Date')['Boarding_Count'].sum().values.reshape(-1, 1)
                    st.session_state['historical_avg'] = daily_data.mean()
                    
                    scaler = MinMaxScaler()
                    scaled_series = scaler.fit_transform(daily_data)
                    
                    win = 14
                    X, y = [], []
                    for i in range(len(scaled_series) - win):
                        X.append(scaled_series[i:i+win])
                        y.append(scaled_series[i+win])
                    
                    X, y = np.array(X), np.array(y)
                    
                    # Build and Train
                    tft = build_tft_model(input_shape=(win, 1))
                    tft.fit(X, y, epochs=10, batch_size=16, verbose=0)
                    
                    # Recursive Forecast
                    preds = []
                    curr_seq = scaled_series[-win:].reshape(1, win, 1)
                    for _ in range(184):
                        p = tft.predict(curr_seq, verbose=0)
                        preds.append(p[0,0])
                        curr_seq = np.append(curr_seq[:, 1:, :], p.reshape(1,1,1), axis=1)
                    
                    forecast_vals = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
                    f_dates = pd.date_range(start="2025-07-01", end="2025-12-31")
                    
                    st.session_state['forecast_df'] = pd.DataFrame({'Date': f_dates, 'Predicted_Demand': forecast_vals.flatten()})
            
            if st.session_state['forecast_df'] is not None:
                f_df = st.session_state['forecast_df']
                
                # Metrics Cards
                m1, m2, m3 = st.columns(3)
                peak_val = f_df['Predicted_Demand'].max()
                avg_val = f_df['Predicted_Demand'].mean()
                growth = ((avg_val - st.session_state['historical_avg']) / st.session_state['historical_avg']) * 100
                
                m1.metric("Predicted Peak Demand", f"{int(peak_val)} Pass.")
                m2.metric("Avg. Forecasted Demand", f"{int(avg_val)} Pass.")
                m3.metric("Projected Growth", f"{growth:.2f}%")

                # Forecast Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_daily['Date'], y=hist_daily['Boarding_Count'], name="Historical"))
                fig.add_trace(go.Scatter(x=f_df['Date'], y=f_df['Predicted_Demand'], name="TFT Forecast", line=dict(dash='dash', color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Prediction Summary Table
                st.subheader("Top 5 Predicted Peak Days")
                st.table(f_df.sort_values(by='Predicted_Demand', ascending=False).head(5))

                # Sidebar Result Display
                match = f_df[f_df['Date'].dt.date == query_date]
                if not match.empty:
                    val = match['Predicted_Demand'].values[0]
                    st.sidebar.success(f"**Prediction for {query_date}:**")
                    st.sidebar.metric("Predicted Boarding", f"{int(val)} Passengers")
                    
                    # Risk Assessment
                    if val > st.session_state['historical_avg'] * 1.3:
                        st.sidebar.error("🚨 CRITICAL: High Overload Risk")
                    elif val > st.session_state['historical_avg'] * 1.15:
                        st.sidebar.warning("⚠️ WARNING: Emerging Imbalance")
                    else:
                        st.sidebar.info("✅ Status: Capacity Normal")

                # Operational Strategy
                st.markdown("---")
                st.subheader("🛠️ Proposed Operational Adjustments")
                if growth > 10:
                    st.write("- **Fleet Reallocation:** Increase peak-hour vehicle frequency by 15% for this route.")
                    st.write("- **Headway Modification:** Reduce headways by 4 minutes during identified peak months.")
                else:
                    st.write("- **Monitoring:** Current capacity is sufficient; maintain standard headways.")

                # Download CSV
                csv = f_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast Data", csv, f"forecast_route_{target_route}.csv", "text/csv")

    else:
        st.error("Please ensure your CSV files contain 'Ridership', 'Traffic', 'Mapping', 'Stops', and 'Routes' in their names.")
else:
    st.info("Waiting for dataset upload (all 5 CSVs required).")