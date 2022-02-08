#!/bin/bash
%env ITERATIONS = 30
%env ACTIVE_LEARNING_BATCHES = 50
%env SEEDS = 398048 127003 259479 869323 570852

%env TREC_TRAIN = /content/project/resources/data/seed_train.csv
%env TREC_TEST= /content/project/resources/data/seed_test.csv
%env AGNEWS_TRAIN = /content/project/resources/data/agnews_train.csv
%env AGNEWS_TEST = /content/project/resources/data/agnews_test.csv

%env MAPPING_PATH = /content/project/resources/mapping/
%env INDICES_PATH = /content/project/resources/indices/
%env CARTOGRAPHY_PATH = /content/project/resources/cartography_plots/
%env RESULTS_PATH = /content/project/results/
%env PLOT_PATH = /content/project/plots/

%env DROPOUT = 0.3
%env EPOCHS = 10
%env EPOCHS_DISCRIMINATIVE = 10
%env MAX_INSTANCE_TREC = 4803
%env MAX_INSTANCE_AGNEWS = 120000
%env MAX_LEN_TREC = 200
%env MAX_LEN_AGNEWS = 200
%env CAL_THRESHOLD = 0.2
# Prepare folders for files to go into
mkdir -p project/{resources/{cartography_plots,embeddings,indices,mapping},results/{agnews,trec},plots/{agnews,trec}}

# Get Data Maps
# Figure 1 -- if you want them annotated -> project/src/utils/save_cartography.py
python3 main.py --task trec --initial_size 5452 --batch_size 16 --pretrained --freeze --cartography --plot
# Figure 2 -- if you want them annotated -> project/src/utils/save_cartography.py
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --cartography --plot
# Full data maps (Appendix)
python3 main.py --task trec --initial_size 5452 --batch_size 16 --pretrained --freeze --cartography --plot --histogram

# Run all acquisition functions for TREC
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition random --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition entropy
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition leastconfidence --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition bald
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition discriminative --analysis
python3 main.py --task trec --initial_size 500 --batch_size 16 --pretrained --freeze --acquisition cartography --analysis

# Plots results in a lineplot (Figure 3) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --initial_size 500 --plot_results

# Run significant tests (Table 2) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --significance

# Run overlapping indices (Table 4) -- Warning: can only be run after all acquisition functions have been ran
python3 main.py --task trec --check_indices
