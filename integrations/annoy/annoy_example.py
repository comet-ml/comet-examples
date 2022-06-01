import random

import comet_ml

from annoy import AnnoyIndex

# Use this if you are using the Cloud version of Comet.ml
# Comment this line if you are using a on-premise version of Comet.ml
comet_ml.init()
# Uncomment this line if you are using a on-premise version of Comet.ml
# comet_ml.init_onprem()

experiment = comet_ml.Experiment()

# Annoy hyper-parameters
f = 40  # Length of item vector that will be indexed
metric = "angular"
seed = 42
output_file = "test.ann"

# Create and fill Annoy Index
t = AnnoyIndex(f, metric)
t.set_seed(seed)

for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10)  # 10 trees

t.save(output_file)

# Comet logging
index_metadata = {
    "f": f,
    metric: metric,
    "n_items": t.get_n_items(),
    "n_trees": t.get_n_trees(),
    "seed": seed,
}

experiment.log_parameters(index_metadata, prefix="annoy_index_1")

experiment.log_asset(output_file, metadata=index_metadata)
