import re


def clean_text(text):

    if not text:
        return ""

    # Xóa khoảng trắng và xuống dòng dư thừa
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng ở đầu và cuối
    text = text.strip()

    return text


if __name__ == "__main__":

    text = """
    Đây     là một     đoạn văn.

    Bài báo      có rất nhiều
    khoảng trắng.
    """

    result = clean_text(text)

    print(result)