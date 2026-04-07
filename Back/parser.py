import re
import pdfplumber
from typing import List


import re
import pdfplumber
from typing import Iterator


def extract_text_pages(pdf_path: str) -> Iterator[str]:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # 1. Удалить номера страниц (вида 109/1738)
                page_text = re.sub(r'\s*\d+/\d+\s*\n?', ' ', page_text)
                # 2. Объединить разорванные строки
                lines = page_text.split('\n')
                merged = []
                for line in lines:
                    if merged and not re.search(r'[.!?:;]\s*$', merged[-1]):
                        merged[-1] += ' ' + line.strip()
                    else:
                        merged.append(line.strip())
                page_text = ' '.join(merged)
                yield page_text
