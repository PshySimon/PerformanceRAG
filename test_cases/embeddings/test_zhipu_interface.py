import os
import pytest
from utils.rag.embedding.zhipu_interface import ZhipuEmbeddings


@pytest.mark.parametrize(
    "text, expected_length",
    [
        ("你好，世界", 1024),  # 假设 embedding-2 返回 1024 维
        ("short", 1024),
    ],
)
def test_embed_text_single(monkeypatch, text, expected_length):
    client = ZhipuEmbeddings()
    # mock 返回固定长度向量
    monkeypatch.setattr(
        client.client.embeddings,
        "create",
        lambda model, input: type(
            "R", (), {"data": [type("D", (), {"embedding": [0.1] * expected_length})]}
        )(),
    )

    vec = client.embed_text(text)
    assert isinstance(vec, list)
    assert len(vec) == expected_length


@pytest.mark.parametrize(
    "texts, expected_lengths",
    [
        (["一", "二", "三"], [1024, 1024, 1024]),
        (["a", "b"], [1024, 1024]),
    ],
)
def test_embed_texts_batch(monkeypatch, texts, expected_lengths):
    client = ZhipuEmbeddings()

    # mock同理
    def fake_create(model, input):
        return type(
            "R",
            (),
            {"data": [type("D", (), {"embedding": [0.2] * 1024}) for _ in input]},
        )()

    monkeypatch.setattr(client.client.embeddings, "create", fake_create)

    vecs = client.embed_texts(texts)
    assert isinstance(vecs, list)
    assert len(vecs) == len(texts)
    for v, exp in zip(vecs, expected_lengths):
        assert isinstance(v, list)
        assert len(v) == exp


def test_real_zhipu_embedding_single():
    """真实调用zhipu embedding接口，单条文本"""
    client = ZhipuEmbeddings()
    text = "你好，世界，zhipu embedding真实调用测试"
    vec = client.embed_text(text)
    print(f"embedding向量长度: {len(vec)}")
    print(f"前10维: {vec[:10]}")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert len(vec) in (1024, 2048)  # embedding-2/3

def test_real_zhipu_embedding_batch():
    """真实调用zhipu embedding接口，批量文本"""
    client = ZhipuEmbeddings()
    texts = ["测试1", "测试2", "zhipu embedding batch"]
    vecs = client.embed_texts(texts)
    print(f"batch返回数量: {len(vecs)}")
    for i, vec in enumerate(vecs):
        print(f"文本{i+1} embedding长度: {len(vec)} 前5维: {vec[:5]}")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) in (1024, 2048)
