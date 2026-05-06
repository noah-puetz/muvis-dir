# Makefile — driver targets for reproducing the per-(method, dataset) table.
#
# Required environment variables:
#   DATA_DIR  Root directory containing the 9 MuViS dataset subdirectories,
#             each with train.ts and test.ts (see README → "Data").
#
# Optional environment variables:
#   SEEDS                Space-separated seeds (default: 42 43 ... 51).
#   OUTPUT_ROOT          Predictions root (default: predictions).
#   PYTHON               Python interpreter (default: python3).

PYTHON       ?= python3
DATA_DIR     ?= /path/to/data
OUTPUT_ROOT  ?= predictions
SEEDS        ?= 42 43 44 45 46 47 48 49 50 51

DATASETS = \
    BeijingPM10Quality \
    BeijingPM25Quality \
    PPGDalia \
    Panasonic18650PFData \
    REVS/2013_Monterey_Motorsports_Reunion \
    REVS/2013_Targa_Sixty_Six \
    REVS/2014_Targa_Sixty_Six \
    TennesseeEastmanProcess \
    VehicleDynamicsDataset

METHODS = lds sqinv focal_l1 conr rnc uvote

.PHONY: help install smoke reproduce clean

help:
	@echo "Targets:"
	@echo "  install    pip install -e . in the current environment"
	@echo "  smoke      train one (small dataset, fast method, single seed) for sanity"
	@echo "  reproduce  walk \$$(METHODS) × \$$(DATASETS) × \$$(SEEDS) = 540 runs"
	@echo ""
	@echo "Required: DATA_DIR=/path/to/MuViS-DIR-data"

install:
	$(PYTHON) -m pip install -e .

# Single-seed smoke test on the smallest dataset (Vehicle Dynamics, 1384 train).
smoke:
	$(PYTHON) -m muvis_dir.train \
	    --dataset VehicleDynamicsDataset \
	    --method  vanilla \
	    --seed    42 \
	    --data-dir $(DATA_DIR) \
	    --output-dir $(OUTPUT_ROOT)/seed_42 \
	    --epochs 5

# 6 methods × 9 datasets × 10 seeds = 540 runs.  Single-GPU loop; for cluster
# execution, replace this body with your scheduler's array submission.
reproduce:
	@for seed in $(SEEDS); do \
	    for ds in $(DATASETS); do \
	        for m in $(METHODS); do \
	            echo "[$$seed][$$ds][$$m]"; \
	            $(PYTHON) -m muvis_dir.train \
	                --dataset "$$ds" \
	                --method "$$m" \
	                --seed "$$seed" \
	                --data-dir $(DATA_DIR) \
	                --output-dir $(OUTPUT_ROOT)/seed_$$seed \
	                || exit 1; \
	        done; \
	    done; \
	done

clean:
	rm -rf predictions build dist *.egg-info
	find src -name __pycache__ -prune -exec rm -rf {} +
