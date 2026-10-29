import re

def nomarlize_text(text):
    text = text.lower()
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )
    return text

def get_words(text):
    text = nomarlize_text(text)
    words = text.split()
    return set(words)

def lexical_score(query, text):
    query_words = get_words(query)
    text_words = get_words(text)

    common_words = query_words & text_words

    score = len(common_words) / len(query_words)

    print("\nQuery words:", query_words)
    print("Common words:", common_words)
    print("Lexical score:", score)

    return score

def rerank_results(
    query,
    results,
    top_k=5
):
    question_type = detect_question_type(query)
    for res in results:

        lexical = lexical_score(
            query,
            res["text"]
        )

        semantic = res["similarity"]
        question_score = question_type_score(question_type,res["text"])
        final_score = (
            0.2 * question_score
            + 0.1 * lexical
            + 0.7 * semantic
        )
        res["question_type"] = question_type
        res["question_type_score"] = question_score
        res["lexical_score"] = lexical
        res["final_score"] = final_score

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results[:top_k]

def detect_question_type(query):
    query = query.lower()
    if(
        "bao nhiêu" in query
        or "mức nào" in query
    ):
        return "NUMERIC"
    if(
        "ở đâu" in query 
        or "tại đâu" in query
        or "địa điểm nào" in query
    ):
        return "LOCATION"
    if(
        "khi nào" in query
        or "bao giờ" in query
        or "thời điểm nào" in query
    ):
        return "DATE"
    if (
        "ai" in query
        or "người nào" in query
    ):
        return "PERSON"

    return "GENERAL"

def question_type_score(question_type, text):
    text = text.lower()

    if question_type == "NUMERIC":
        # Kiểm tra chunk có số hay không
        money_pattern = r"""
        \d+(?:[.,]\d+)?
        \s*
        (
            tỷ\s*usd
            |triệu\s*usd
            |usd
            |đô\s*la
            |tỷ\s*đồng
            |triệu\s*đồng
            |nghìn\s*tỷ
            |tỷ\s*vnđ
            |triệu\s*vnđ
        )
    """

        if re.search(
            money_pattern,
            text,
            re.IGNORECASE | re.VERBOSE
        ):
            return 1.0
        return 0.0

    if question_type == "LOCATION":
        location_keywords = [
        "hà nội",
        "hồ chí minh",
        "tp.hcm",
        "thành phố hồ chí minh",
        "bắc ninh",
        "thái nguyên",
        "hải phòng",
        "đà nẵng",
        "quảng ninh",
        "hưng yên",
        "thanh hóa",
        "nghệ an",
        "kiên giang",
        "cần thơ"
    ]

        for word in location_keywords:
            if word in text:
                return 1.0

        return 0.0

    if question_type == "DATE":
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"\d{1,2}-\d{1,2}-\d{4}",
            r"\bnăm \d{4}\b",
            r"\btháng \d{1,2}\b"
        ]

        for pattern in date_patterns:
            if re.search(pattern, text):
                return 1.0

        return 0.0

    if question_type == "PERSON":
        person_keywords = [
            "ông ",
            "bà ",
            "anh ",
            "chị ",
            "ông",
            "bà"
        ]

        for word in person_keywords:
            if word in text:
                return 1.0

        return 0.0

    return 0.0 
        
def merge_adjacent_chunks(results):
    merged = []
    used = set()
    for i , result in enumerate(results):
        if i in used :
            continue 
        current = result.copy()
        for j , next_result in enumerate(results):
            if j==i or j in used :
                continue
            same_article = next_result["article_id"] == current["article_id"]
            adjacent_chunk = abs(next_result["chunk_id"] - current["chunk_id"]) == 1
            if same_article and adjacent_chunk :
                if(next_result["chunk_id"]>current["chunk_id"]):
                    current["text"] = current["text"] + " " + next_result["text"]
                else : 
                    current["text"] = next_result["text"] + " " +current["text"]
                used.add(j)
    return merged
            


if __name__ == "__main__":
    questions = [
        "Samsung đầu tư bao nhiêu tiền tại Việt Nam?",
        "Samsung đầu tư ở đâu tại Việt Nam?",
        "Sự kiện này diễn ra khi nào?",
        "Ai là người phát biểu?"
    ]

    test_texts = [
        "Samsung đã đầu tư 24 tỷ USD tại Việt Nam.",
        "Samsung có các nhà máy tại Bắc Ninh và Thái Nguyên.",
        "Sự kiện diễn ra vào năm 2025.",
        "Ông Nguyễn Văn A phát biểu tại hội nghị."
    ]

    for question in questions:

        question_type = detect_question_type(question)

        print("\nCâu hỏi:", question)
        print("Loại:", question_type)

        for text in test_texts:
            score = question_type_score(
                question_type,
                text
            )

            print("Text:", text)
            print("Question type score:", score)