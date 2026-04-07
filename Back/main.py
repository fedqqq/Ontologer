import sqlite3
from retrieval import PDFVectorIndexer
from generator import Generator
from tester import TestingModule


def build_context(results: list) -> str:
    context_parts = []
    for page_num, chunk_idx, chunk_text, score in results:
        context_parts.append(f"[Страница {page_num}, релевантность {score:.2%}]\n{chunk_text}\n")
    return "\n".join(context_parts).strip()


def consultation_mode(indexer, generator):
    print("\n--- Режим консультации ---")
    print("Задавайте вопросы по материалу. Для выхода введите 'exit'.\n")
    while True:
        user_query = input("Ваш вопрос: ").strip()
        if user_query.lower() in ("exit", "quit", "q"):
            break
        if not user_query:
            continue

        print("\nПоиск релевантных чанков...")
        results = indexer.search_similar(user_query, top_k=5)
        if not results:
            print("Не найдено подходящих фрагментов в документе.\n")
            continue

        context = build_context(results)
        print("\n" + "=" * 80)
        print("КОНТЕКСТ, ПЕРЕДАННЫЙ В МОДЕЛЬ:")
        print("=" * 80)
        print(context)
        print("=" * 80 + "\n")

        print("Генерация ответа...\n")
        answer = generator.generate(context, user_query)
        print(f"Ответ: {answer}\n")
        print("-" * 60)


def testing_mode(tester):
    print("\n--- Режим тестирования ---")
    print("Будут сгенерированы вопросы на основе блоков последовательных чанков.")
    print("Для каждого вопроса вам нужно будет дать ответ.")
    print("Введите 'skip', чтобы пропустить вопрос.\n")

    # Запускаем сессию
    tester.run_test_session()


def main():
    pdf_path = "DM2024-сжатый.pdf"
    db_path = "ontologer.db"
    model_path = "T-lite-it-2.1-Q4_K_M.gguf"

    # 1. Индексатор
    print("Инициализация индексатора...")
    indexer = PDFVectorIndexer(
        pdf_path=pdf_path,
        db_path=db_path,
        resume=True,
        model_name="intfloat/multilingual-e5-large"
    )

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM chunk_embeddings")
        count = cur.fetchone()[0]
    if count == 0:
        print("База данных пуста, начинаю индексацию...")
        indexer.process()
    else:
        print(f"БД уже содержит {count} чанков. Пропускаем индексацию.")

    # 2. Генератор LLM
    print("Загрузка LLM модели...")
    generator = Generator(model_path=model_path)

    # 3. Модуль тестирования
    tester = TestingModule(
        indexer=indexer,
        generator=generator,
        db_path="testing.db",
        chunks_per_question=10,      # 10 последовательных чанков на вопрос
        questions_per_session=5       # 5 вопросов за сессию
    )

    # 4. Главное меню
    while True:
        print("\n" + "=" * 50)
        print("ГЛАВНОЕ МЕНЮ")
        print("1. Консультация (задать вопрос по документу)")
        print("2. Тестирование (проверить свои знания)")
        print("3. Показать историю тестов")
        print("4. Выйти")
        print("=" * 50)
        choice = input("Выберите режим (1/2/3/4): ").strip()

        if choice == "1":
            consultation_mode(indexer, generator)
        elif choice == "2":
            testing_mode(tester)
        elif choice == "3":
            tester.show_statistics()
        elif choice == "4":
            print("До свидания!")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите 1, 2, 3 или 4.")


if __name__ == "__main__":
    main()