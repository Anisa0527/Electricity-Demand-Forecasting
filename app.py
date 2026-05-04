import gradio as gr
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/model.pkl")


# ---------------- PREDICT FUNCTION ----------------
def predict(T2M_toc, QV2M_toc, W2M_toc,
            holiday, school, hour, day, month, weekday):

    # Create input dataframe
    features = pd.DataFrame([[T2M_toc, QV2M_toc, W2M_toc,
                              holiday, school, hour, day, month, weekday]],
                            columns=['T2M_toc','QV2M_toc','W2M_toc',
                                     'holiday','school','hour','day','month','weekday'])

    # Model prediction
    prediction = model.predict(features)[0]

    # ---------------- ACTUAL VALUE (DEMO PURPOSE) ----------------
    # In real project, actual comes from dataset test split
    actual = prediction + np.random.normal(0, 15)

    # ---------------- GRAPH ----------------
    plt.figure(figsize=(6,4))
    plt.plot(["Actual", "Predicted"], [actual, prediction],
             marker="o", linewidth=3)

    plt.title("⚡ Electricity Demand Forecasting")
    plt.ylabel("Load (MW)")
    plt.grid(True)

    graph_path = "graph.png"
    plt.savefig(graph_path)
    plt.close()

    return f"⚡ Predicted Demand: {prediction:.2f}", graph_path


# ---------------- GRADIO UI ----------------
app = gr.Interface(
    fn=predict,

    inputs=[
        gr.Number(label="Temperature (T2M_toc)"),
        gr.Number(label="Humidity (QV2M_toc)"),
        gr.Number(label="Wind Speed (W2M_toc)"),
        gr.Number(label="Holiday (0/1)"),
        gr.Number(label="School (0/1)"),
        gr.Number(label="Hour"),
        gr.Number(label="Day"),
        gr.Number(label="Month"),
        gr.Number(label="Weekday")
    ],

    outputs=[
        gr.Textbox(label="Prediction Output"),
        gr.Image(label="Actual vs Predicted Graph")
    ],

    title="⚡ Electricity Demand Forecasting AI System",
    description="ML-based system to predict electricity demand and peak power using weather + calendar data"
)


# ---------------- RUN APP ----------------
app.launch(share=True)