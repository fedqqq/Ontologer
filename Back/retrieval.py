import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Tuple
from parser import extract_text_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFVectorIndexer:
    def __init__(
            self,
            pdf_path: str,
            db_path: str = "vectors.db",
            model_name: str = "all-MiniLM-L6-v2",
            batch_size: int = 1,
            resume: bool = True,
            min_text_length: int = 100,
            chunk_size: int = 500,  # размер чанка в символах
            chunk_overlap: int = 50  # перекрытие между чанками
    ):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.resume = resume
        self.min_text_length = min_text_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Инициализируем сплиттер для текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        self._init_db()

    def _init_db(self):
        """Создаём таблицу для хранения чанков, а не страниц целиком."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_number INTEGER,
                    chunk_index INTEGER,          -- порядковый номер чанка на странице
                    chunk_text TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Создаём индекс для быстрого поиска по странице
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page ON chunk_embeddings(page_number)")

    def _is_page_processed(self, page_num: int) -> bool:
        """Проверяем, были ли уже сохранены чанки для данной страницы."""
        if not self.resume:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM chunk_embeddings WHERE page_number = ? LIMIT 1",
                (page_num,)
            )
            return cur.fetchone() is not None

    def _embed_text(self, text: str) -> bytes:
        """Преобразует текст в бинарный эмбеддинг."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32).tobytes()

    def _save_chunk(self, page_num: int, chunk_index: int, chunk_text: str, embedding_blob: bytes):
        """Сохраняет один чанк в базу данных."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chunk_embeddings (page_number, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                (page_num, chunk_index, chunk_text, embedding_blob)
            )

    def process(self):
        """Основной метод: читает страницы, разбивает на чанки, индексирует."""
        page_generator = extract_text_pages(self.pdf_path)
        page_num = 1

        while True:
            try:
                text = next(page_generator)
            except StopIteration:
                break

            if self._is_page_processed(page_num):
                print(f"Страница {page_num} уже обработана (её чанки существуют), пропускаем.")
                page_num += 1
                continue

            stripped_text = text.strip()
            if len(stripped_text) < self.min_text_length:
                print(f"Страница {page_num} содержит недостаточно текста (длина: {len(stripped_text)}), пропускаем.")
                page_num += 1
                continue

            print(f"Обработка страницы {page_num} (разбиение на чанки размером {self.chunk_size})...")

            # Разбиваем текст страницы на чанки
            chunks = self.text_splitter.split_text(stripped_text)

            if not chunks:
                print(f"Страница {page_num} не разбилась на чанки (пустой текст после очистки).")
                page_num += 1
                continue

            # Индексируем каждый чанк
            for idx, chunk in enumerate(chunks):
                embedding_blob = self._embed_text(chunk)
                self._save_chunk(page_num, idx, chunk, embedding_blob)

            print(f"Страница {page_num} сохранена (чанков: {len(chunks)}).")
            page_num += 1

        print("Обработка завершена.")

    def get_chunks_by_page(self, page_num: int) -> List[Tuple[int, str, np.ndarray]]:
        """Возвращает все чанки для указанной страницы: (chunk_index, текст, эмбеддинг)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT chunk_index, chunk_text, embedding FROM chunk_embeddings WHERE page_number = ? ORDER BY chunk_index",
                (page_num,)
            )
            results = []
            for idx, text, blob in cur.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                results.append((idx, text, emb))
            return results

    def get_page_text(self, page_num: int) -> Optional[str]:
        """Восстанавливает полный текст страницы, склеивая чанки (без учёта перекрытий, просто порядок)."""
        chunks = self.get_chunks_by_page(page_num)
        if not chunks:
            return None
        # Склеиваем чанки в исходном порядке (перекрытия могут привести к дублированию, но для большинства задач достаточно)
        full_text = "\n".join([text for _, text, _ in chunks])
        return full_text

    def search_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[int, int, str, float]]:
        """
        Ищет топ-k наиболее релевантных чанков.
        Возвращает список: (page_number, chunk_index, chunk_text, similarity_score)
        """
        query_emb = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT page_number, chunk_index, chunk_text, embedding FROM chunk_embeddings")
            results = []
            for page_num, chunk_idx, chunk_text, blob in cur.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                results.append((page_num, chunk_idx, chunk_text, sim))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    # Для обратной совместимости можно оставить старые методы, но они будут работать некорректно.
    # Рекомендую их удалить или переопределить с предупреждением.
    def get_embedding(self, page_num: int) -> None:
        raise NotImplementedError("Этот метод больше не поддерживается. Используйте get_chunks_by_page().")

    def get_text(self, page_num: int) -> None:
        raise NotImplementedError("Этот метод больше не поддерживается. Используйте get_page_text().")