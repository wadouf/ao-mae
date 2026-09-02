PYTHONPATH := $(CURDIR)/src
export PYTHONPATH

preflight:
	python scripts/00_preflight.py --config config/project.yaml

test:
	pytest -q

run:
	python scripts/run_pipeline.py --config config/project.yaml

text-audit:
	python scripts/audit_package_text.py .
