PYTHONPATH := $(CURDIR)/src
export PYTHONPATH

CONFIG ?= config/project.yaml
CITY   ?= dak
K      ?= 25
SEED   ?= 1

preflight:
	python scripts/00_preflight.py --config $(CONFIG)

test:
	pytest -q

pretrain:
	python scripts/20_pretrain_stage1.py --config $(CONFIG) --held-out-city $(CITY)

train:
	python scripts/21_train_stage2.py --config $(CONFIG) \
	  --pretrained outputs/checkpoints/stage1/stage1_encoders_$(CITY)_seed01.pt \
	  --held-out-city $(CITY) --few-shot-k $(K) --seed $(SEED)

benchmark:
	python scripts/22_benchmark_compute.py --config $(CONFIG)

values:
	python manuscript/scripts/build_values.py

run:
	python scripts/run_pipeline.py --config $(CONFIG)

text-audit:
	python scripts/audit_package_text.py .
