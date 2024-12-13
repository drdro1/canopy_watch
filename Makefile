#################### ACTIONS ###################
run_api:
	uvicorn canopywatch.api.fast:app --reload

run_preprocess_train_evaluate:
	python -c 'from canopywatch.ml_logic.model import evaluate; evaluate()'

run_prediction:
	python -c 'from canopywatch.ml_logic.model import prediction; prediction()'

run_streamlit:
	-@streamlit run streamlit/app.py
