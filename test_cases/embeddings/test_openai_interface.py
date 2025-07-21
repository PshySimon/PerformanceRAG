import os
import pytest
import openai

from utils.rag.embedding.openai_interface import OpenAIEmbeddingsClient


# 默认环境变量设置
@pytest.fixture(autouse=True)
def ensure_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    yield


@pytest.fixture
def client():
    return OpenAIEmbeddingsClient()


@pytest.mark.parametrize(
    "text, expected_len",
    [
        ("hello world", 3),  # 伪设 embedding 长度为3
        ("单条测试", 5),
    ],
)
def test_embed_text_single(monkeypatch, client, text, expected_len):
    # Mock OpenAI 返回
    fake_resp = {"data": [{"embedding": [0.1] * expected_len}]}
    monkeypatch.setattr(openai.Embedding, "create", lambda model, input: fake_resp)

    vec = client.embed_text(text)
    assert isinstance(vec, list)
    assert len(vec) == expected_len


@pytest.mark.parametrize(
    "texts, expected_lens",
    [
        (["a", "b", "c"], [4, 4, 4]),
        (["短", "文本"], [2, 2]),
    ],
)
def test_embed_texts_batch(monkeypatch, client, texts, expected_lens):
    # 模拟 batch 返回
    def fake_create(model, input):
        return {
            "data": [
                {"embedding": [0.2] * expected_len} for expected_len in expected_lens
            ]
        }

    monkeypatch.setattr(openai.Embedding, "create", fake_create)

    vecs = client.embed_texts(texts)
    assert isinstance(vecs, list)
    assert len(vecs) == len(texts)
    for vec, exp in zip(vecs, expected_lens):
        assert isinstance(vec, list)
        assert len(vec) == exp
