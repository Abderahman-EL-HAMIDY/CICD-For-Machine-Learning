install:
	pip install --upgrade pip
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

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	pip install -U huggingface_hub
	python -m huggingface_hub.commands.huggingface_cli login --token $(HF) --add-to-git-credential

push-hub:
	python -m huggingface_hub.commands.huggingface_cli upload Abderahman-el-hamidy/Weather-Prediction ./App --repo-type=space --commit-message="Sync App files"
	python -m huggingface_hub.commands.huggingface_cli upload Abderahman-el-hamidy/Weather-Prediction ./Model /Model --repo-type=space --commit-message="Sync Model"
	python -m huggingface_hub.commands.huggingface_cli upload Abderahman-el-hamidy/Weather-Prediction ./Results /Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
