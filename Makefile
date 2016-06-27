.PHONY=venv
venv::
	source ./venv/bin/activate

.PHONY=frontend
frontend::
	(cd frontend && python -m SimpleHTTPServer 8000)

.PHONY=backend
backend::
	python runbackend.py