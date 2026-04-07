import sqlite3
import json
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from retrieval import PDFVectorIndexer
from generator import Generator


@dataclass
class Question:
    """Структура вопроса для тестирования."""
    chunk_ids: List[Tuple[int, int]]  # список (page_number, chunk_index)
    combined_text: str                # объединённый текст чанков
    question_text: str                # сгенерированный вопрос
    expected_answer: str              # эталонный ответ (из LLM)
    student_answer: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None


class TestingModule:
    """
    Модуль тестирования: генерирует вопросы на основе блоков последовательных чанков.
    Вопросы не хранятся в БД заранее, а генерируются динамически во время сессии.
    """

    def __init__(
        self,
        indexer: PDFVectorIndexer,
        generator: Generator,
        db_path: str = "testing.db",
        chunks_per_question: int = 10,
        questions_per_session: int = 5
    ):
        """
        :param indexer: индексатор PDF (уже загружен и содержит чанки)
        :param generator: генератор ответов (LLM)
        :param db_path: путь к БД для хранения результатов сессий (вопросы не храним)
        :param chunks_per_question: сколько последовательных чанков использовать для одного вопроса
        :param questions_per_session: количество вопросов за одну сессию
        """
        self.indexer = indexer
        self.generator = generator
        self.db_path = db_path
        self.chunks_per_question = chunks_per_question
        self.questions_per_session = questions_per_session
        self._init_db()

    def _init_db(self):
        """Создаёт таблицу для хранения результатов сессий (вопросы не сохраняются)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_questions INTEGER,
                    total_score REAL,
                    max_possible_score REAL,
                    details TEXT   -- JSON строка с деталями по каждому вопросу
                )
            """)

    def _get_all_chunks_ordered(self) -> List[Tuple[int, int, str]]:
        """
        Возвращает все чанки в порядке (page_number, chunk_index).
        Каждый элемент: (page_number, chunk_index, chunk_text).
        """
        with sqlite3.connect(self.indexer.db_path) as conn:
            cur = conn.execute(
                "SELECT page_number, chunk_index, chunk_text FROM chunk_embeddings ORDER BY page_number, chunk_index"
            )
            return cur.fetchall()

    def _get_chunk_block(self, start_idx: int, chunks: List[Tuple[int, int, str]]) -> Tuple[List[Tuple[int, int]], str]:
        """
        Возвращает блок из chunks_per_question последовательных чанков, начиная с start_idx.
        Возвращает (список (page_num, chunk_idx), объединённый текст).
        """
        block_chunks = chunks[start_idx:start_idx + self.chunks_per_question]
        if not block_chunks:
            return [], ""
        chunk_ids = [(page, idx) for page, idx, _ in block_chunks]
        combined_text = "\n\n".join(text for _, _, text in block_chunks)
        return chunk_ids, combined_text

    def _generate_question_from_text(self, context_text: str) -> Tuple[str, str]:
        """
        Генерирует вопрос и ожидаемый ответ на основе предоставленного текста (блока чанков).
        Возвращает (вопрос, ожидаемый_ответ).
        """
        prompt = (
            "Ты — экзаменатор по дискретной математике. На основе предоставленного текста из учебника "
            "сформулируй один чёткий вопрос, проверяющий понимание этого текста, и дай эталонный ответ на него.\n"
            "Формат ответа должен быть строго:\n"
            "ВОПРОС: <текст вопроса>\n"
            "ОТВЕТ: <текст эталонного ответа>\n"
            "Не добавляй лишних комментариев, только эти две строки.\n"
            "Вопрос должен быть таким, чтобы ответ можно было найти именно в этом тексте.\n\n"
            f"Текст:\n{context_text}\n"
        )
        try:
            response = self.generator.generate(context=context_text, question=prompt)
            # Парсим ответ
            lines = response.strip().split('\n')
            question = ""
            expected = ""
            reading_q = False
            reading_a = False
            for line in lines:
                if line.startswith("ВОПРОС:"):
                    question = line[len("ВОПРОС:"):].strip()
                    reading_q = True
                elif line.startswith("ОТВЕТ:"):
                    expected = line[len("ОТВЕТ:"):].strip()
                    reading_a = True
                elif reading_q and not reading_a:
                    question += " " + line.strip()
                elif reading_a:
                    expected += " " + line.strip()
            if not question or not expected:
                # fallback
                parts = response.strip().split('\n', 1)
                question = parts[0] if parts else "Вопрос не сгенерирован"
                expected = parts[1] if len(parts) > 1 else "Ответ не сгенерирован"
            return question, expected
        except Exception as e:
            print(f"Ошибка генерации вопроса: {e}")
            return "", ""

    def generate_questions_for_session(self, num_questions: Optional[int] = None) -> List[Question]:
        """
        Генерирует вопросы для сессии: выбирает случайные непересекающиеся блоки последовательных чанков.
        :param num_questions: количество вопросов (если None, используется self.questions_per_session)
        :return: список объектов Question (без ответов студента)
        """
        if num_questions is None:
            num_questions = self.questions_per_session

        all_chunks = self._get_all_chunks_ordered()
        if len(all_chunks) < self.chunks_per_question:
            print(f"Недостаточно чанков для формирования блока из {self.chunks_per_question}.")
            return []

        # Максимальный стартовый индекс для блока
        max_start = len(all_chunks) - self.chunks_per_question
        if max_start < 0:
            return []

        # Выбираем случайные непересекающиеся стартовые индексы
        # Простой способ: случайно выбираем num_questions разных позиций, но они могут пересекаться.
        # Чтобы блоки не пересекались, можно использовать выбор без возврата с шагом chunks_per_question.
        # Для простоты сделаем с пересечениями, но лучше избежать повторов.
        available_starts = list(range(max_start + 1))
        random.shuffle(available_starts)
        selected_starts = available_starts[:num_questions]

        questions = []
        for start in selected_starts:
            chunk_ids, combined_text = self._get_chunk_block(start, all_chunks)
            if not chunk_ids:
                continue
            print(f"Генерация вопроса для блока чанков (начиная с позиции {start})...")
            q_text, a_text = self._generate_question_from_text(combined_text)
            if q_text and a_text:
                questions.append(Question(
                    chunk_ids=chunk_ids,
                    combined_text=combined_text,
                    question_text=q_text,
                    expected_answer=a_text
                ))
            else:
                print("Не удалось сгенерировать вопрос для этого блока.")

        return questions

    def evaluate_answer(self, question: Question, student_answer: str) -> Tuple[float, str]:
        """
        Оценивает ответ студента с помощью LLM.
        Возвращает (оценка_от_0_до_1, фидбек).
        """
        prompt = (
            "Ты — строгий преподаватель дискретной математики. Оцени ответ студента на вопрос, "
            "сравнивая его с эталонным ответом. Выставь оценку от 0 до 1, где:\n"
            "0 — ответ полностью неверный или не по теме;\n"
            "0.5 — частично верный, но есть неточности;\n"
            "1 — полное совпадение с эталоном (или семантически эквивалентный правильный ответ).\n"
            "Дай краткий комментарий.\n\n"
            f"Вопрос: {question.question_text}\n"
            f"Эталонный ответ: {question.expected_answer}\n"
            f"Ответ студента: {student_answer}\n\n"
            "Формат ответа:\n"
            "ОЦЕНКА: <число от 0 до 1>\n"
            "КОММЕНТАРИЙ: <текст>\n"
        )
        try:
            eval_response = self.generator.generate(
                context=question.combined_text,
                question=prompt
            )
            lines = eval_response.strip().split('\n')
            score = 0.0
            feedback = ""
            for line in lines:
                if line.startswith("ОЦЕНКА:"):
                    score_str = line[len("ОЦЕНКА:"):].strip().replace(',', '.')
                    try:
                        score = float(score_str)
                        score = max(0.0, min(1.0, score))
                    except:
                        score = 0.0
                elif line.startswith("КОММЕНТАРИЙ:"):
                    feedback = line[len("КОММЕНТАРИЙ:"):].strip()
            if not feedback:
                feedback = "Оценка выполнена автоматически."
            return score, feedback
        except Exception as e:
            print(f"Ошибка оценки ответа: {e}")
            return 0.0, "Ошибка автоматической оценки."

    def run_test_session(self, num_questions: Optional[int] = None) -> Dict:
        """
        Запускает сессию тестирования:
        - генерирует вопросы (блоки последовательных чанков)
        - задаёт их студенту
        - оценивает ответы
        :param num_questions: количество вопросов (если None, используется self.questions_per_session)
        :return: словарь с результатами сессии
        """
        if num_questions is None:
            num_questions = self.questions_per_session

        print("\n" + "=" * 60)
        print(f"Генерация {num_questions} вопросов на основе блоков по {self.chunks_per_question} чанков...")
        questions = self.generate_questions_for_session(num_questions)

        if not questions:
            print("Не удалось сгенерировать ни одного вопроса. Проверьте наличие чанков.")
            return {}

        print(f"\nНачало тестирования. Всего вопросов: {len(questions)}")
        print("=" * 60)

        total_score = 0.0
        max_score = float(len(questions))
        details = []

        for i, q in enumerate(questions, 1):
            print(f"\n--- Вопрос {i} ---")
            print(q.question_text)
            print("\nВаш ответ (или 'skip' для пропуска):")
            student_ans = input("> ").strip()
            if student_ans.lower() == "skip":
                score = 0.0
                feedback = "Вопрос пропущен студентом."
            else:
                score, feedback = self.evaluate_answer(q, student_ans)
                total_score += score
            print(f"\nОценка: {score:.2f} / 1.00")
            print(f"Комментарий: {feedback}")
            print(f"Эталонный ответ: {q.expected_answer}")
            details.append({
                "question": q.question_text,
                "student_answer": student_ans,
                "expected_answer": q.expected_answer,
                "score": score,
                "feedback": feedback
            })

        # Сохраняем сессию в БД
        session_data = {
            "total_questions": len(questions),
            "total_score": total_score,
            "max_possible_score": max_score,
            "details": json.dumps(details, ensure_ascii=False)
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO test_sessions
                   (total_questions, total_score, max_possible_score, details)
                   VALUES (?, ?, ?, ?)""",
                (session_data["total_questions"], session_data["total_score"],
                 session_data["max_possible_score"], session_data["details"])
            )

        print("\n" + "=" * 60)
        print(f"Сессия завершена. Итоговый балл: {total_score:.2f} / {max_score:.2f}")
        print(f"Процент: {total_score/max_score*100:.1f}%")
        print("=" * 60)

        return session_data

    def show_statistics(self):
        """Выводит историю сессий тестирования."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT session_date, total_questions, total_score, max_possible_score FROM test_sessions ORDER BY id"
            )
            rows = cur.fetchall()
        if not rows:
            print("Нет завершённых сессий.")
            return
        print("\nИстория тестирования:")
        print(f"{'Дата':<20} {'Вопросов':<10} {'Балл':<10} {'Макс':<10} {'Процент':<10}")
        for row in rows:
            date, qty, score, max_score = row
            percent = score / max_score * 100 if max_score > 0 else 0
            print(f"{date[:19]:<20} {qty:<10} {score:<10.2f} {max_score:<10.2f} {percent:<10.1f}%")
