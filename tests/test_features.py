"""data.one_hot / features.kmer_counts 的已知答案测试（CI CPU 上跑）。"""

from enhancerscope.data import one_hot, to_codes
from enhancerscope.features import gc_fraction, kmer_counts, kmer_features, kmer_vocab


def test_one_hot_known_sequence() -> None:
    out = one_hot(["ACGT"])
    assert out.shape == (1, 4, 4)
    assert out[0, :, :].tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_one_hot_non_acgt_is_zero_column() -> None:
    out = one_hot(["ANNT"])
    assert out[0, :, 1].sum() == 0  # N 位置全零
    assert out[0, :, 0].sum() == 1 and out[0, :, 3].sum() == 1


def test_to_codes_values() -> None:
    codes = to_codes(["AC", "GT"])
    assert codes.tolist() == [[0, 1], [2, 3]]


def test_kmer_counts_known() -> None:
    mat = kmer_counts(["AAAC"], k=2)
    dense = mat.toarray()[0]
    # 2-mer: AA x2, AA? 窗口为 AA/AA/AC → AA=2, AC=1；无其他
    assert dense[0] == 2.0  # AA (索引 0)
    assert dense[1] == 1.0  # AC (索引 1)
    assert dense.sum() == 3.0  # 4-2+1=3 个窗口


def test_kmer_vocab_size_and_order() -> None:
    vocab = kmer_vocab(2)
    assert len(vocab) == 16
    assert vocab[:3] == ["AA", "AC", "AG"]


def test_gc_fraction() -> None:
    gc = gc_fraction(["GGCC", "AAAA"])
    assert gc.tolist() == [1.0, 0.0]


def test_kmer_features_shape() -> None:
    mat = kmer_features(["ACGTACGT", "TTTTGGGG"], k=3)
    assert mat.shape == (2, 4**3 + 1)  # 64 k-mer + 1 GC
    assert abs(mat[0, -1] - 0.5) < 1e-6  # ACGTACGT: C,G,C,G = 4/8
    assert abs(mat[1, -1] - 0.5) < 1e-6  # TTTTGGGG: 4/8
