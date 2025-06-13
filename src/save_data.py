import os
os.environ['XDG_CACHE_HOME'] = '/tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/relbench-ignacio/data'

root_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'relbench')
data_name = 'f1'
task = 'u'

import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from relbench.modeling.utils import get_stype_proposal
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader


from torch import Tensor


from models import HeteroGraphGIN

# Some book keeping
from torch_geometric.seed import seed_everything
seed_everything(42)



dataset = get_dataset(f"rel-{data_name}", download=True)
task = get_task(f"rel-{data_name}", f"{task}", download=True)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # check that it's cuda if you want it to run in reasonable time!

db = dataset.get_db()
col_to_stype_dict = get_stype_proposal(db)
col_to_stype_dict

 
class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))



text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device=device), batch_size=256
)

data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,  # speficied column types
    text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
    cache_dir=os.path.join(
        root_dir, f"rel-{data_name}", f"rel-{data_name}_materialized_cache"
    ),  # store materialized graph for convenience
)

loader_dict = {}

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    table_input = get_node_train_table_input(
        table=table,
        task=task,
    )
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[
            32 for i in range(2)
        ],  # we sample subgraphs of depth 2, 128 neighbors per node.
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=128,
        temporal_strategy="uniform",
        shuffle=split == "train",
        num_workers=0,
        persistent_workers=False,
    )


torch.save(data, os.path.join(root_dir, f"rel-{data_name}", f"rel-{data_name}-graph.pth"))
torch.save(col_stats_dict, os.path.join(root_dir, f"rel-{data_name}", f"rel-{data_name}-col_stats_dict.pth"))
 