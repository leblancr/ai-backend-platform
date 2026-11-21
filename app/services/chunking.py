def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 150,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind("\n\n", start, end)

            if boundary == -1 or boundary < start + chunk_size // 2:
                boundary = text.rfind("\n", start, end)

            if boundary == -1 or boundary < start + chunk_size // 2:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        # Move the next starting point to a word boundary.
        while next_start < end and text[next_start] != " " and text[next_start] != "\n":
            next_start += 1

        start = next_start

    return chunks