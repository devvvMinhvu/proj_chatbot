def split_text(text, chunk_size=400, overlap=50):
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải từ 0 đến nhỏ hơn chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Đã xử lý đến cuối văn bản thì dừng.
        if end >= len(words):
            break

        start = end - overlap

    return chunks