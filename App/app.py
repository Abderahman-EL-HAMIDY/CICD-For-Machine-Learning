import gradio as gr
import skops.io as sio
import pandas as pd
# Load the trained pipeline
pipe = sio.load("../Model/weather_pipeline.skops", trusted=['numpy.dtype'])


def predict_rain(temperature, humidity, wind_speed, cloud_cover, pressure):
    features = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed,
        "Cloud_Cover": cloud_cover,
        "Pressure": pressure
    }])
    predicted_rain = pipe.predict(features)[0]
    return f"Prediction: {predicted_rain}"



# Create input components
inputs = [
    gr.Slider(0, 50, step=0.1, label="Temperature (°C)", value=25),
    gr.Slider(0, 100, step=0.1, label="Humidity (%)", value=50),
    gr.Slider(0, 30, step=0.1, label="Wind Speed (km/h)", value=10),
    gr.Slider(0, 100, step=0.1, label="Cloud Cover (%)", value=50),
    gr.Slider(980, 1050, step=0.1, label="Pressure (hPa)", value=1013),
]

outputs = [gr.Label(num_top_classes=2)]

# Example inputs for testing
examples = [
    [31.4, 74.3, 10.4, 68.3, 997.2],  # Example 1
    [14.7, 83.3, 10.4, 13.9, 986.4],  # Example 2
    [10.8, 64.8, 15.0, 18.0, 1013.4], # Example 3
]

title = "Weather Prediction - Rain Classifier"
description = "Enter weather conditions to predict if it will rain or not."
article = "This app is part of a CI/CD pipeline for Machine Learning. It demonstrates automated training, evaluation, and deployment using GitHub Actions."

# Create and launch the interface
gr.Interface(
    fn=predict_rain,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch(share=True)
