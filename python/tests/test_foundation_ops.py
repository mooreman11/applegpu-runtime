import numpy as np
import pytest
import applegpu_runtime as gpu


@pytest.fixture(autouse=True)
def init():
    gpu.init_backend()


def test_scalar_mul():
    t = gpu.tensor([2.0, 4.0, 6.0], shape=[3])
    result = gpu.scalar_mul(t, 0.5).to_list()
    assert abs(result[0] - 1.0) < 0.01
    assert abs(result[1] - 2.0) < 0.01


def test_reshape_roundtrip():
    t = gpu.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6])
    t2 = gpu.reshape(t, [2, 3])
    assert t2.shape == [2, 3]
    assert t2.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_slice_multihead():
    # Simulate multi-head: [2, 128] -> slice into 2 heads of [2, 64]
    data = list(range(256))
    t = gpu.tensor([float(x) for x in data], shape=[2, 128])
    head0 = gpu.slice(t, dim=1, start=0, end=64)
    head1 = gpu.slice(t, dim=1, start=64, end=128)
    assert head0.shape == [2, 64]
    assert head1.shape == [2, 64]


def test_concat_multihead():
    a = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    b = gpu.tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    c = gpu.concat(a, b, dim=1)
    assert c.shape == [2, 4]
    result = c.to_list()
    assert result == [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]


def test_add_bias_eval():
    x = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    bias = gpu.tensor([10.0, 20.0], shape=[2])
    result = gpu.add_bias(x, bias).to_list()
    assert abs(result[0] - 11.0) < 0.01  # 1+10
    assert abs(result[1] - 22.0) < 0.01  # 2+20
    assert abs(result[2] - 13.0) < 0.01  # 3+10
    assert abs(result[3] - 24.0) < 0.01  # 4+20


def test_softmax_causal_masking():
    # 3x3 input, causal mask should zero out future positions
    data = [1.0] * 9
    t = gpu.tensor(data, shape=[3, 3])
    result = np.array(gpu.softmax_causal(t).to_list()).reshape(3, 3)
    # Row 0: only position 0, so [1.0, 0.0, 0.0]
    assert abs(result[0, 0] - 1.0) < 0.01
    assert abs(result[0, 1]) < 0.01
    assert abs(result[0, 2]) < 0.01
    # Row 1: positions 0,1, so [0.5, 0.5, 0.0]
    assert abs(result[1, 0] - 0.5) < 0.01
    assert abs(result[1, 1] - 0.5) < 0.01
    assert abs(result[1, 2]) < 0.01


def test_attention_causal_eval():
    q = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    k = gpu.tensor([1.0, 0.0, 0.0, 1.0], shape=[2, 2])
    v = gpu.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    result = gpu.attention_causal(q, k, v).to_numpy()
    assert result.shape == (2, 2)
    # Row 0 can only attend to position 0
    assert abs(result[0, 0] - 1.0) < 0.2
    assert abs(result[0, 1] - 2.0) < 0.2


def test_argmax_eval():
    t = gpu.tensor([1.0, 5.0, 3.0, 2.0, 8.0, 4.0], shape=[2, 3])
    result = gpu.argmax(t)
    assert result.dtype == "int32"
    indices = result.to_list()
    assert indices[0] == 1  # max of [1, 5, 3] is at index 1
    assert indices[1] == 1  # max of [2, 8, 4] is at index 1


def test_argmax_dtype():
    t = gpu.tensor([1.0, 2.0, 3.0], shape=[1, 3])
    result = gpu.argmax(t)
    assert result.dtype == "int32"
    assert result.to_list() == [2]


def test_gpt2_attention_block():
    """Full multi-head attention pattern: project -> slice -> attention x N -> concat"""
    seq_len = 4
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads  # 4

    # Input and weight matrix
    x = gpu.from_numpy(np.random.randn(seq_len, d_model).astype(np.float32))
    w_qkv = gpu.from_numpy(np.random.randn(d_model, d_model * 3).astype(np.float32))

    # Project Q, K, V in one matmul
    qkv = x @ w_qkv  # [seq, d_model*3]

    # Split into Q, K, V
    q_all = gpu.slice(qkv, dim=1, start=0, end=d_model)
    k_all = gpu.slice(qkv, dim=1, start=d_model, end=d_model * 2)
    v_all = gpu.slice(qkv, dim=1, start=d_model * 2, end=d_model * 3)

    # Per-head attention
    head_outputs = []
    for h in range(n_heads):
        q_h = gpu.slice(q_all, dim=1, start=h * d_head, end=(h + 1) * d_head)
        k_h = gpu.slice(k_all, dim=1, start=h * d_head, end=(h + 1) * d_head)
        v_h = gpu.slice(v_all, dim=1, start=h * d_head, end=(h + 1) * d_head)
        out_h = gpu.attention_causal(q_h, k_h, v_h)
        head_outputs.append(out_h)

    # Concat heads
    result = head_outputs[0]
    for h in head_outputs[1:]:
        result = gpu.concat(result, h, dim=1)

    output = result.to_numpy()
    assert output.shape == (seq_len, d_model)
    assert np.all(np.isfinite(output))
