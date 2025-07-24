import time

from flask import Flask, jsonify, request

from utils.config import config
from rag.components.embedding.hf_embedding import CustomHFEmbedding

app = Flask(__name__)

# 缓存模型实例，避免重复加载
MODEL_CACHE = {}


def get_embedding_model(model_name=None):
    if not model_name:
        model_name = config.embeddings.clients.hf.model_name
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = CustomHFEmbedding(model_name=model_name)
    return MODEL_CACHE[model_name]


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    data = request.get_json(force=True)
    input_data = data.get("input")
    model_name = data.get("model")
    if input_data is None:
        return jsonify({"error": {"message": "Missing 'input' field"}}), 400
    if isinstance(input_data, str):
        texts = [input_data]
    elif isinstance(input_data, list):
        texts = input_data
    else:
        return (
            jsonify(
                {"error": {"message": "'input' must be string or list of strings"}}
            ),
            400,
        )
    model = get_embedding_model(model_name)
    start = time.time()
    try:
        embeddings = model.embed_texts(texts)
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
    duration = time.time() - start
    # openai格式输出
    resp = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": model_name or config.embeddings.clients.hf.model_name,
        "usage": {"prompt_tokens": None, "total_tokens": None, "duration": duration},
    }
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
