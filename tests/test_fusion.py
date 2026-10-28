
from app.retrieval import Hit, rrf_fuse


def make(cid, dense_rank=None, sparse_rank=None):
    return Hit(id=cid, text=cid, source="s", heading="", page=None, strategy="recursive",
               dense_rank=dense_rank, sparse_rank=sparse_rank)


def test_chunk_in_both_lists_wins():
    dense = [make("a", dense_rank=1), make("b", dense_rank=2)]
    sparse = [make("b", sparse_rank=1), make("c", sparse_rank=2)]
    fused = rrf_fuse(dense, sparse, 0.5, 0.5)
    assert fused[0].id == "b"


def test_weights_change_order():
    dense = [make("a", dense_rank=1)]
    sparse = [make("c", sparse_rank=1)]
    assert rrf_fuse(dense, sparse, 0.9, 0.1)[0].id == "a"
    dense = [make("a", dense_rank=1)]
    sparse = [make("c", sparse_rank=1)]
    assert rrf_fuse(dense, sparse, 0.1, 0.9)[0].id == "c"