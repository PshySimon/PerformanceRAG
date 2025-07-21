from flask import Flask, request, Response
from utils.rag.pipeline import create_pipeline
import json

app = Flask(__name__)
pipeline = create_pipeline("advanced_rag")


@app.route("/chat", methods=["POST"])
def rag_endpoint():
    data = request.get_json(force=True)
    query = data.get("query")
    if not query:
        return (
            Response(
                json.dumps({"error": "Missing 'query' parameter"}, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
            ),
            400,
        )
    try:
        result = pipeline.run(query)
        return Response(
            json.dumps({"result": result}, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return (
            Response(
                json.dumps({"error": str(e)}, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
