install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	@echo "Uploading to Hugging Face Space..."
	huggingface-cli upload Abderahman-el-hamidy/Weather-Prediction ./App/weather_app.py weather_app.py --repo-type space --space-sdk streamlit
	huggingface-cli upload Abderahman-el-hamidy/Weather-Prediction ./App/requirements.txt requirements.txt --repo-type space --space-sdk streamlit

deploy: hf-login push-hub
