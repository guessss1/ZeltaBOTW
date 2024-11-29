import os
import numpy as np
import openai
from dotenv import load_dotenv
from pypdf import PdfReader
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import redis
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Загрузка переменных из .env
load_dotenv()

# Чтение переменных из .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))  # По умолчанию порт 6379
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Инициализация OpenAI
openai.api_key = OPENAI_API_KEY

# Константы для работы с Redis
INDEX_NAME = "embeddings-index"
PREFIX = "doc"
DISTANCE_METRIC = "COSINE"


class DataService:
    def __init__(self):
        # Подключение к Redis
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True  # Декодирование строк в человеко-читаемый формат
            )
            # Проверка соединения
            if self.redis_client.ping():
                logging.info("Успешное подключение к Redis.")
        except Exception as e:
            logging.error(f"Ошибка подключения к Redis: {e}")
            raise

    def drop_redis_data(self, index_name: str = INDEX_NAME):
        """Удаление индекса в Redis (если он существует)."""
        try:
            self.redis_client.ft(index_name).dropindex()
            logging.info("Индекс удалён.")
        except Exception as e:
            logging.warning(f"Индекс не существует или не может быть удалён: {e}")

    def load_data_to_redis(self, embeddings):
        """Загрузка данных (эмбеддингов) в Redis."""
        try:
            vector_dim = len(embeddings[0]['vector'])
            vector_number = len(embeddings)

            # Определение полей индекса
            text = TextField(name="text")
            text_embedding = VectorField("vector",
                                         "FLAT", {
                                             "TYPE": "FLOAT32",
                                             "DIM": vector_dim,
                                             "DISTANCE_METRIC": DISTANCE_METRIC,
                                             "INITIAL_CAP": vector_number,
                                         })
            fields = [text, text_embedding]

            # Проверка существования индекса
            try:
                self.redis_client.ft(INDEX_NAME).info()
                logging.info("Индекс уже существует.")
            except:
                # Создание нового индекса
                self.redis_client.ft(INDEX_NAME).create_index(
                    fields=fields,
                    definition=IndexDefinition(
                        prefix=[PREFIX], index_type=IndexType.HASH)
                )
                logging.info(f"Индекс '{INDEX_NAME}' создан.")

            # Загрузка данных в Redis
            for embedding in embeddings:
                key = f"{PREFIX}:{str(embedding['id'])}"
                embedding["vector"] = np.array(
                    embedding["vector"], dtype=np.float32).tobytes()
                self.redis_client.hset(key, mapping=embedding)
            logging.info(f"Загружено {self.redis_client.info()['db0']['keys']} документов в Redis.")
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных в Redis: {e}")
            raise

    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 1000):
        """Чтение PDF, разбиение на чанки и создание эмбеддингов через OpenAI."""
        try:
            reader = PdfReader(pdf_path)
            chunks = []
            for page in reader.pages:
                text_page = page.extract_text()
                chunks.extend([text_page[i:i + chunk_length].replace('\n', '')
                               for i in range(0, len(text_page), chunk_length)])

            # Создание эмбеддингов через OpenAI
            response = openai.Embedding.create(model='text-embedding-ada-002', input=chunks)
            logging.info(f"Создано {len(response['data'])} эмбеддингов из PDF.")
            return [{'id': idx, 'vector': item['embedding'], 'text': chunks[idx]} for idx, item in
                    enumerate(response['data'])]
        except Exception as e:
            logging.error(f"Ошибка при обработке PDF или создании эмбеддингов: {e}")
            raise

    def search_redis(self,
                     user_query: str,
                     index_name: str = INDEX_NAME,
                     vector_field: str = "vector",
                     return_fields: list = ["text", "vector_score"],
                     hybrid_fields="*",
                     k: int = 5,
                     print_results: bool = False,
                     ):
        """Поиск по Redis с использованием KNN и запроса пользователя."""
        try:
            # Создание эмбеддинга для пользовательского запроса
            embedded_query = openai.Embedding.create(
                input=user_query, model="text-embedding-ada-002")['data'][0]['embedding']

            # Определение запроса
            base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
            query = (
                Query(base_query)
                .return_fields(*return_fields)
                .sort_by("vector_score")
                .paging(0, k)
                .dialect(2)
            )
            params_dict = {"vector": np.array(
                embedded_query).astype(dtype=np.float32).tobytes()}

            # Выполнение поиска
            results = self.redis_client.ft(index_name).search(query, params_dict)
            if print_results:
                for i, doc in enumerate(results.docs):
                    score = 1 - float(doc.vector_score)
                    logging.info(f"{i}. {doc.text} (Score: {round(score, 3)})")
            return [doc['text'] for doc in results.docs]
        except Exception as e:
            logging.error(f"Ошибка при поиске в Redis: {e}")
            raise

