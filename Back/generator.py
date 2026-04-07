import logging
from typing import Optional
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self._default_prompt()

    @staticmethod
    def _default_prompt():
        return (
            "Ты — строгий ассистент, который отвечает ТОЛЬКО на основе предоставленного контекста. "
            "Если в контексте нет точного ответа на вопрос, ты ОБЯЗАН ответить: 'Не знаю, в контексте нет такой информации.' "
            "НЕ используй свои знания, НЕ додумывай, НЕ обобщай. "
            "Отвечай максимально кратко и только фактами из контекста. "
            "Если контекст не относится к вопросу, также отвечай 'Не знаю...'"
        )

    def generate(self, context: str, question: str) -> str:
        """
        Генерирует ответ на вопрос, используя переданный контекст.
        Поддерживает как chat-шаблон (через create_chat_completion), так и ручной формат.
        """
        # Попытка использовать стандартный chat-шаблон (работает для большинства современных GGUF)
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
            ]
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["<|im_end|>", "<|endoftext|>"],   # стандартные стоп-токены
            )
            answer = response["choices"][0]["message"]["content"].strip()
            if answer:
                return answer
            # если пусто — пробуем fallback
            logger.warning("create_chat_completion вернул пустой ответ, пробуем ручной формат")
        except Exception as e:
            logger.warning(f"create_chat_completion не удался: {e}, переключаюсь на ручной формат")

        # Ручной формат для моделей, не поддерживающих chat-шаблон (например, старые или специфичные)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            f"Ответ:"
        )
        try:
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["\n\nВопрос:", "\nКонтекст:", "<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
            answer = response["choices"][0]["text"].strip()
            return answer if answer else "Не удалось сгенерировать ответ."
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при генерации ответа."
