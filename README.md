#  Weather Prediction - CI/CD for Machine Learning

A complete CI/CD pipeline for a Machine Learning project that predicts rain based on weather conditions. This project demonstrates automated training, evaluation, and deployment using GitHub Actions and Hugging Face Spaces.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![Gradio](https://img.shields.io/badge/Gradio-4.44.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Live Demo

**Try the app here:** [Weather Prediction on Hugging Face](https://huggingface.co/spaces/Abderahman-el-hamidy/Weather-Prediction)

##  Project Overview

This project implements an end-to-end ML pipeline that:

- Trains a Random Forest classifier to predict rain based on weather features
- Automatically evaluates model performance and generates reports
- Deploys the model as a web application on Hugging Face Spaces
- Uses GitHub Actions for continuous integration and deployment

##  Project Structure
```
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration workflow
│       └── cd.yml              # Continuous Deployment workflow
├── App/
│   ├── app.py                  # Gradio web application
│   ├── requirements.txt        # App dependencies
│   └── README.md               # Hugging Face Space config
├── Data/
│   └── data.csv                # Training dataset
├── Model/
│   └── weather_pipeline.skops  # Trained model pipeline
├── Results/
│   ├── metrics.txt             # Model performance metrics
│   └── model_results.png       # Confusion matrix plot
├── Makefile                    # Automation commands
├── requirements.txt            # Project dependencies
├── train.py                    # Model training script
└── README.md                   # This file
```

##  Features

### Input Features
| Feature | Description | Range |
|---------|-------------|-------|
| Temperature | Temperature in Celsius | 0 - 50°C |
| Humidity | Humidity percentage | 0 - 100% |
| Wind Speed | Wind speed in km/h | 0 - 30 km/h |
| Cloud Cover | Cloud cover percentage | 0 - 100% |
| Pressure | Atmospheric pressure in hPa | 980 - 1050 hPa |

### Output
- **Prediction:** Rain or No Rain

##  Tech Stack

- **Machine Learning:** scikit-learn, pandas, numpy
- **Model Serialization:** skops
- **Web App:** Gradio
- **CI/CD:** GitHub Actions
- **Deployment:** Hugging Face Spaces

##  Installation

### Clone the repository
```bash
git clone https://github.com/Abderahman-EL-HAMIDY/CICD-For-Machine-Learning.git
cd CICD-For-Machine-Learning
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Run the app locally
```bash
cd App
python app.py
```

##  CI/CD Pipeline

### Continuous Integration (CI)
Triggered on push/pull request to main branch:
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Train model
5. Evaluate and generate reports

### Continuous Deployment (CD)
Triggered after CI completes:
1. Install Hugging Face CLI
2. Login to Hugging Face
3. Upload App, Model, and Results to Hugging Face Space

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| F1 Score | 1.0 |

##  Usage

### Using the Web App
1. Visit the [Hugging Face Space](https://huggingface.co/spaces/Abderahman-el-hamidy/Weather-Prediction)
2. Adjust the sliders for weather conditions
3. Click "Submit" to get the prediction

### Using the Model Programmatically
```python
import skops.io as sio
import pandas as pd

# Load the model
pipe = sio.load("Model/weather_pipeline.skops", trusted=['numpy.dtype'])

# Make prediction
features = pd.DataFrame([{
    "Temperature": 25.0,
    "Humidity": 80.0,
    "Wind_Speed": 15.0,
    "Cloud_Cover": 70.0,
    "Pressure": 1000.0
}])

prediction = pipe.predict(features)[0]
print(f"Prediction: {prediction}")
```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Abderahman EL-HAMIDY**

- GitHub: [@Abderahman-EL-HAMIDY](https://github.com/Abderahman-EL-HAMIDY)
- Hugging Face: [@Abderahman-el-hamidy](https://huggingface.co/Abderahman-el-hamidy)

##  Acknowledgments

- This project was built following the CI/CD for Machine Learning tutorial
- Thanks to Hugging Face for providing free hosting for ML demos
- Built with scikit-learn and Gradio

---

