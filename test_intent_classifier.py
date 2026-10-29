
from nlp.intent_classifier import (
    predict_intent,
    predict_intent_with_confidence,
)


test_cases = [
    "Chào bạn nhé",
    "Hệ thống của bạn có thể làm gì?",
    "Cho tôi xem các tin vừa đăng",
    "Hãy tóm lược nội dung bài viết",
    "Tôi muốn đọc bài báo gốc",
    "Tra cứu các bài báo về học máy",
    "Dự án vệ sinh học đường có những hoạt động nào?",
]

for text in test_cases:
    result = predict_intent_with_confidence(text)

    print(f"\nCâu hỏi: {text}")
    print(f"Ý định: {result['intent']}")
    print(f"Xác suất: {result['confidence']:.2%}")
