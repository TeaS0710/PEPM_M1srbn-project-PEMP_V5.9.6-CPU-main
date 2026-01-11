SHELL := /bin/bash

PY := .venv/bin/python

# ---- User parameters -----------------------------------------------------
PROFILE ?= ideo_quick
STAGE   ?= pipeline

# Optional execution controls (forwarded as overrides)
CORPUS_ID        ?=
DATASET_ID       ?=
FAMILY           ?=
HARDWARE_PRESET  ?=
TRAIN_PROP       ?=
OVERRIDES        ?=

# ---- Helpers -------------------------------------------------------------

define ADD_OVERRIDE
$(if $(strip $(2)),--override $(1)=$(2),)
endef

OVR = \
  $(call ADD_OVERRIDE,corpus_id,$(CORPUS_ID)) \
  $(call ADD_OVERRIDE,dataset_id,$(DATASET_ID)) \
  $(call ADD_OVERRIDE,families,$(FAMILY)) \
  $(call ADD_OVERRIDE,hardware_preset,$(HARDWARE_PRESET)) \
  $(call ADD_OVERRIDE,train_prop,$(TRAIN_PROP)) \
  $(foreach o,$(OVERRIDES),--override $(o))

# ---- Targets -------------------------------------------------------------

.PHONY: help setup init_dirs sysinfo prepare train evaluate pipeline run audit_tei audit_splits aggregate clean

help:
	@echo "PEPM v5.9.6 (CPU) – minimal Makefile"
	@echo
	@echo "Core runs:"
	@echo "  make run STAGE=pipeline PROFILE=ideo_quick"
	@echo "  make run STAGE=pipeline PROFILE=ideo_quick_web1_asr1"
	@echo "  make run STAGE=pipeline PROFILE=custom   OVERRIDES='view=crawl label_field=crawl'"
	@echo
	@echo "Useful audits:"
	@echo "  make audit_tei   CORPUS_ID=web1"
	@echo "  make audit_splits DIR=data/interim/<dataset>/<view>"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

init_dirs:
	mkdir -p data/raw/web1 data/raw/asr1 data/interim data/processed models reports

sysinfo:
	$(PY) scripts/tools/sysinfo.py --out_json reports/sysinfo.json

prepare:
	$(PY) scripts/core/core_prepare.py --profile $(PROFILE) $(OVR) --verbose

train:
	$(PY) scripts/core/core_train.py --profile $(PROFILE) $(OVR) --verbose

evaluate:
	$(PY) scripts/core/core_evaluate.py --profile $(PROFILE) $(OVR) --verbose

pipeline: prepare train evaluate

run: init_dirs
	@if [ "$(STAGE)" = "prepare" ]; then \
	  $(MAKE) prepare; \
	elif [ "$(STAGE)" = "train" ]; then \
	  $(MAKE) train; \
	elif [ "$(STAGE)" = "evaluate" ]; then \
	  $(MAKE) evaluate; \
	elif [ "$(STAGE)" = "pipeline" ]; then \
	  $(MAKE) pipeline; \
	else \
	  echo "[ERROR] STAGE must be one of: prepare | train | evaluate | pipeline"; exit 2; \
	fi

# Audits (optional)
DIR ?=
OUT_JSON ?=

audit_tei:
	$(PY) scripts/tools/audit_tei.py data/raw/$(CORPUS_ID)/corpus.xml --out_json $(if $(OUT_JSON),$(OUT_JSON),reports/audit_tei_$(CORPUS_ID).json)

audit_splits:
	$(PY) scripts/tools/audit_splits.py $(DIR) --out_json $(if $(OUT_JSON),$(OUT_JSON),reports/audit_splits.json)

aggregate:
	$(PY) scripts/post/post_aggregate_metrics.py --reports_dir reports --out_tsv reports/summary.tsv

clean:
	rm -rf data/interim/* data/processed/* models/*
	@echo "[clean] OK"
