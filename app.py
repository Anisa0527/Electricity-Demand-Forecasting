import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# LOAD DATA
# =====================================================

data = pd.read_csv("data/archive/continuous dataset.csv")

data['datetime'] = pd.to_datetime(data['datetime'])

# =====================================================
# LOAD RANDOM FOREST MODEL
# =====================================================

rf_model = joblib.load("models/model.pkl")

# =====================================================
# LOAD LSTM MODEL
# =====================================================

lstm_model = load_model("lstm_model.h5")

# =====================================================
# SCALER FOR LSTM
# =====================================================

dataset = data['nat_demand'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))

dataset_scaled = scaler.fit_transform(dataset)

# =====================================================
# MAIN DASHBOARD FUNCTION
# =====================================================

def electricity_dashboard(
    T2M_toc,
    QV2M_toc,
    W2M_toc,
    holiday,
    school,
    hour,
    day,
    month,
    weekday
):

    # =================================================
    # RANDOM FOREST PREDICTION
    # =================================================

    features = pd.DataFrame([[

        T2M_toc,
        QV2M_toc,
        W2M_toc,
        holiday,
        school,
        hour,
        day,
        month,
        weekday

    ]], columns=[

        'T2M_toc',
        'QV2M_toc',
        'W2M_toc',
        'holiday',
        'school',
        'hour',
        'day',
        'month',
        'weekday'

    ])

    rf_prediction = rf_model.predict(features)[0]

    # =================================================
    # LSTM 24-HOUR FORECAST
    # =================================================

    future_predictions = []

    last_24 = dataset_scaled[-24:].flatten().tolist()

    temp_input = last_24.copy()

    for i in range(24):

        x_input = np.array(temp_input[-24:])
        x_input = x_input.reshape(1,24,1)

        yhat = lstm_model.predict(x_input, verbose=0)

        temp_input.append(yhat[0][0])

        prediction = scaler.inverse_transform(
            yhat.reshape(-1,1)
        )[0][0]

        future_predictions.append(prediction)

    # =================================================
    # PEAK / AVERAGE / MINIMUM DEMAND
    # =================================================

    peak_demand = data['nat_demand'].max()

    average_demand = data['nat_demand'].mean()

    minimum_demand = data['nat_demand'].min()

    # =================================================
    # AI INSIGHTS
    # =================================================

    if rf_prediction > average_demand:
        insight = "⚠ High electricity demand expected."
    else:
        insight = "✅ Electricity demand is within normal range."

    # =================================================
    # ACTUAL VS PREDICTED GRAPH
    # =================================================

    sample = data.tail(100)

    actual = sample['nat_demand'].values

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual Demand'
    ))

    fig1.add_trace(go.Scatter(
        y=np.append(actual[:-1], rf_prediction),
        mode='lines',
        name='Predicted Demand'
    ))

    fig1.update_layout(
        title="⚡ Actual vs Predicted Electricity Demand",
        xaxis_title="Time",
        yaxis_title="Demand",
        template="plotly_dark"
    )

    # =================================================
    # 24-HOUR FORECAST GRAPH
    # =================================================

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        y=future_predictions,
        mode='lines+markers',
        name='24 Hour Forecast'
    ))

    fig2.update_layout(
        title="🧠 LSTM 24-Hour Future Forecast",
        xaxis_title="Next 24 Hours",
        yaxis_title="Predicted Demand",
        template="plotly_dark"
    )

    # =================================================
    # RESULT TEXT
    # =================================================

    result = f"""

⚡ RANDOM FOREST DEMAND PREDICTION:
{rf_prediction:.2f}

🧠 LSTM NEXT HOUR FORECAST:
{future_predictions[0]:.2f}

🔥 PEAK POWER DEMAND:
{peak_demand:.2f}

📊 AVERAGE DEMAND:
{average_demand:.2f}

📉 MINIMUM DEMAND:
{minimum_demand:.2f}

🤖 AI INSIGHT:
{insight}

"""

    return result, fig1, fig2

# =====================================================
# PROFESSIONAL DASHBOARD UI
# =====================================================

theme = gr.themes.Soft()

demo = gr.Interface(

    fn=electricity_dashboard,

    inputs=[

        gr.Number(label="🌡 Temperature"),
        gr.Number(label="💧 Humidity"),
        gr.Number(label="🌬 Wind Speed"),
        gr.Number(label="🎉 Holiday (0/1)"),
        gr.Number(label="🏫 School (0/1)"),
        gr.Number(label="⏰ Hour"),
        gr.Number(label="📅 Day"),
        gr.Number(label="📆 Month"),
        gr.Number(label="🗓 Weekday")

    ],

    outputs=[

        gr.Textbox(label="⚡ AI Power System Analysis"),

        gr.Plot(label="📊 Actual vs Predicted Graph"),

        gr.Plot(label="🧠 24-Hour Forecast Dashboard")

    ],

    title="⚡ Intelligent Electricity Load & Peak Demand Forecasting System",

    description="""
AI-based Electricity Demand Forecasting using:

✅ Random Forest Machine Learning
✅ LSTM Deep Learning
✅ Peak Power Demand Analysis
✅ 24-Hour Future Forecasting
✅ Interactive Dashboard Analytics
""",

    theme=theme
)

demo.launch(share=True)
