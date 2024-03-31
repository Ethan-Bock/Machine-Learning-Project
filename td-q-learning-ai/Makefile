# Expects to be used when a Python environment has been activated
VENV := .env

# if the virtual environment has not been activated
ifeq ($(VIRTUAL_ENV),)
all install: $(VENV) 
	@echo "Remember to start your virtual environment with "
	@echo "  source $(VENV)/bin/activate"

$(VENV):
	@echo "You need to make the virtual environment: "
	@echo "    python3 -m venv $(VENV)"


# if the virtual environment has been activated
else
all:
	@echo make install - install all python packages 
	@echo make demo    - run the demo with one set of options

install:
	pip3 install -r requirements.txt

demo:
	./td-q-learning-ai.py  learn --environment lake --model-file demo-model.joblib --alpha 0.5 --gamma 0.8 --epsilon-chance-factor 0.25 --n-epochs 100 --track-epochs 1 --track-steps 0
	./td-q-learning-ai.py score --environment lake --model-file demo-model.joblib

endif