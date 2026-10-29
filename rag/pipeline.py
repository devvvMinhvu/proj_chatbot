from nlp.vector_store import semantic_search_raw
from processing.reranking import rerank_results

def retrieve_chunks(query, candidate_k=20, context_k=5):

    candidates = semantic_search_raw(
        query,
        k=candidate_k
    )

    results = rerank_results(
    query,
    candidates,
    top_k=context_k
)

    print("\n========== RETRIEVAL ==========")
    print("Query:", query)

    for i, result in enumerate(results):

        print(f"\n--- Chunk {i + 1} ---")
        print("Article ID:", result["article_id"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", result["similarity"])
        print("Lexical:", result["lexical_score"])
        print("Question type : ",result["question_type"])
        print("Question score: ",result["question_type_score"])
        print("Final score:", result["final_score"])
        print("Text:")
        print(result["text"][:500])

    return results


def build_context(results):

    context = ""

    for i, result in enumerate(results):

        context += f"""
[Đoạn thông tin {i + 1}]
Article ID: {result["article_id"]}

Nội dung:
{result["text"]}
"""

    return context


if __name__ == "__main__":

    query = input(
        "Nhập câu hỏi: "
    )

    results = retrieve_chunks(
        query,
        candidate_k=20,
        context_k=10
    )

    context = build_context(
        results
    )

    print(
        "\n\n========== CONTEXT =========="
    )

    print(context)