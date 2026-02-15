# 🔧 CRITICAL UPDATE V2.1 - Детальный разбор с примерами кода

**Дата:** 21 января 2026  
**Версия:** 2.1  
**Время на чтение:** 15 минут  
**Для кого:** Backend-разработчики, AI-ассистенты (CLINE)

---

## 📖 СОДЕРЖАНИЕ

1. [Проблема и решение](#проблема-и-решение)
2. [Миграция на Google Gemini](#миграция-на-google-gemini)
3. [Изменение размерности](#изменение-размерности)
4. [Новая архитектура поиска](#новая-архитектура-поиска)
5. [Взвешенная релевантность](#взвешенная-релевантность)
6. [Система фильтров](#система-фильтров)
7. [Примеры полного кода](#примеры-полного-кода)

---

## 🎯 ПРОБЛЕМА И РЕШЕНИЕ

### Текущее состояние MVP (❌)

```python
# backend/embeddings.py - ТЕКУЩИЙ КОД (НЕПРАВИЛЬНО)
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self):
        # Использует локальную mini-LLM
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # ❌ НЕПРАВИЛЬНАЯ РАЗМЕРНОСТЬ
    
    def embed_text(self, text: str) -> np.ndarray:
        """Создает эмбеддинг размерностью 384"""
        return self.model.encode(text)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Создает batch эмбеддингов 384"""
        return self.model.encode(texts)
```

**Проблемы:**
1. ❌ Использует локальную модель (sentence-transformers) вместо Google Gemini
2. ❌ Размерность - чтобы везде 384 
3. ❌ Нет взвешивания по полям
4. ❌ Смешанный поиск (семантика + ключевые слова)

### Целевое состояние (✅)

```python
# backend/embeddings.py - НОВЫЙ КОД (ПРАВИЛЬНО)
import google.generativeai as genai
import numpy as np
import os
from typing import List

class EmbeddingService:
    def __init__(self):
        # Подключаем Google Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY не найден в .env")
        
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"
        self.dimension = 384  # ✅ ПРАВИЛЬНАЯ РАЗМЕРНОСТЬ
    
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """
        Создает эмбеддинг размерностью 384 через Google Gemini API
        
        Args:
            text: Текст для эмбеддинга
            task_type: Тип задачи (retrieval_document или retrieval_query)
        
        Returns:
            numpy array размерностью 384
        """
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type
        )
        return np.array(result['embedding'])
    
    def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Создает batch эмбеддингов размерностью 384
        
        Args:
            texts: Список текстов
            task_type: Тип задачи
        
        Returns:
            numpy array формы (len(texts), 384)
        """
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            embeddings.append(result['embedding'])
        
        return np.array(embeddings)
```

---

## 🔄 МИГРАЦИЯ НА GOOGLE GEMINI

### Шаг 1: Установка зависимостей

**Удалить старые:**
```bash
pip uninstall -y sentence-transformers transformers torch torchvision torchaudio
```

**Установить новые:**
```bash
pip install google-generativeai python-dotenv
```

**Обновить requirements.txt:**
```txt
# Старое (удалить):
# sentence-transformers==2.2.2
# transformers==4.30.0
# torch==2.0.1

# Новое (добавить):
google-generativeai==0.3.1
python-dotenv==1.0.0
numpy==1.24.3
scikit-learn==1.3.0
```

### Шаг 2: Получение API Key

1. Перейти на https://aistudio.google.com
2. Нажать "Get API Key"
3. Создать новый API Key или использовать существующий
4. Скопировать ключ

### Шаг 3: Настройка .env

Создать файл `.env` в корне проекта:
```bash
# .env
GOOGLE_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxx

# Другие переменные
DATABASE_URL=postgresql://user:pass@localhost/db
DEBUG=True
```

**Добавить .env в .gitignore:**
```bash
echo ".env" >> .gitignore
```

### Шаг 4: Загрузка переменных окружения

```python
# backend/config.py
from dotenv import load_dotenv
import os

# Загрузить переменные из .env
load_dotenv()

# Конфигурация
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    EMBEDDING_DIMENSION = 384  # ✅ ВСЕГДА 384
    
    @classmethod
    def validate(cls):
        """Проверить наличие обязательных переменных"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY отсутствует в .env файле")
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL отсутствует в .env файле")

# Вызвать валидацию при импорте
Config.validate()
```

---

## 📐 ИЗМЕНЕНИЕ РАЗМЕРНОСТИ

### Где изменить 384 → 384

#### 1. Инициализация массивов эмбеддингов

**Было:**
```python
# ❌ СТАРЫЙ КОД
embeddings = np.zeros((7283, 384))
query_embedding = np.zeros(384)
```

**Стало:**
```python
# ✅ НОВЫЙ КОД
embeddings = np.zeros((7283, 384))
query_embedding = np.zeros(384)
```

#### 2. Загрузка эмбеддингов из файла

**Было:**
```python
# ❌ СТАРЫЙ КОД
def load_embeddings(file_path: str) -> np.ndarray:
    """Загрузка эмбеддингов 384"""
    embeddings = np.load(file_path)
    assert embeddings.shape[1] == 384, "Неправильная размерность"
    return embeddings
```

**Стало:**
```python
# ✅ НОВЫЙ КОД
def load_embeddings(file_path: str) -> np.ndarray:
    """Загрузка эмбеддингов 384"""
    embeddings = np.load(file_path)
    assert embeddings.shape[1] == 384, f"Ожидается 384, получено {embeddings.shape[1]}"
    return embeddings
```

#### 3. Создание таблицы в БД (PostgreSQL с pgvector)

**Было:**
```sql
-- ❌ СТАРАЯ СХЕМА
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)  -- ❌ НЕПРАВИЛЬНО
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

**Стало:**
```sql
-- ✅ НОВАЯ СХЕМА
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    FAS_arguments_embedding vector(384),      -- ✅ ПРАВИЛЬНО
    violation_summary_embedding vector(384),  -- ✅ ПРАВИЛЬНО
    addescription_embedding vector(384)       -- ✅ ПРАВИЛЬНО
);

-- Индексы для быстрого поиска
CREATE INDEX ON documents USING ivfflat (FAS_arguments_embedding vector_cosine_ops);
CREATE INDEX ON documents USING ivfflat (violation_summary_embedding vector_cosine_ops);
CREATE INDEX ON documents USING ivfflat (addescription_embedding vector_cosine_ops);
```

#### 4. Пересоздание эмбеддингов

**Скрипт миграции:**
```python
# scripts/migrate_embeddings.py
import numpy as np
from backend.embeddings import EmbeddingService
from backend.database import Database
from tqdm import tqdm

def migrate_embeddings():
    """
    Пересоздать все эмбеддинги с размерностью 384
    """
    print("🔄 Начинаем миграцию эмбеддингов...")
    
    # Инициализация
    embedding_service = EmbeddingService()
    db = Database()
    
    # Получить все документы
    documents = db.get_all_documents()
    print(f"📊 Найдено {len(documents)} документов")
    
    # Создать новые эмбеддинги
    for doc in tqdm(documents, desc="Создание эмбеддингов"):
        # Три поля для эмбеддинга
        fields = {
            'FAS_arguments': doc['FAS_arguments'],
            'violation_summary': doc['violation_summary'],
            'addescription': doc['addescription']
        }
        
        # Создать эмбеддинги для каждого поля
        embeddings = {}
        for field_name, field_text in fields.items():
            if field_text and field_text.strip():
                emb = embedding_service.embed_text(
                    field_text,
                    task_type="retrieval_document"
                )
                embeddings[f'{field_name}_embedding'] = emb
            else:
                # Если поле пустое, нулевой вектор
                embeddings[f'{field_name}_embedding'] = np.zeros(384)
        
        # Обновить в БД
        db.update_embeddings(doc['id'], embeddings)
    
    print("✅ Миграция завершена!")

if __name__ == "__main__":
    migrate_embeddings()
```

---

## 🔍 НОВАЯ АРХИТЕКТУРА ПОИСКА

### Старая архитектура (❌)

```python
# ❌ СТАРЫЙ ПОДХОД (НЕПРАВИЛЬНО)
def search_old(query: str, filters: dict) -> list:
    """
    Старая архитектура: фильтры ДО поиска
    """
    # 1. Применить фильтры СРАЗУ
    filtered_docs = apply_filters(all_documents, filters)
    
    # 2. Семантический поиск по отфильтрованным
    query_embedding = embedding_service.embed_text(query)
    semantic_results = vector_search(query_embedding, filtered_docs)
    
    # 3. Если мало результатов - добавить ключевые слова
    if len(semantic_results) < 10:
        keyword_results = keyword_search(query, filtered_docs)
        semantic_results.extend(keyword_results)
    
    # 4. Вернуть топ-10-20
    return semantic_results[:20]
```

**Проблемы:**
- Фильтры убивают релевантные документы до поиска
- Смешивание семантики и ключевых слов усложняет логику
- Непредсказуемое количество результатов (10-20)

### Новая архитектура (✅)

```python
# ✅ НОВЫЙ ПОДХОД (ПРАВИЛЬНО)
def search_new(query: str, filters: dict) -> list:
    """
    Новая архитектура: TOP-50 → фильтры → взвешивание → TOP-10
    """
    # 1. Создать эмбеддинг запроса
    query_embedding = embedding_service.embed_text(
        query,
        task_type="retrieval_query"  # Важно: query, не document!
    )
    
    # 2. Семантический поиск по ВСЕМ документам → TOP-50
    top_50_candidates = vector_search_top50(query_embedding)
    
    # 3. Применить фильтры к TOP-50
    filtered_candidates = apply_filters(top_50_candidates, filters)
    
    # 4. Взвешенная сортировка по трем полям
    scored_results = calculate_weighted_scores(
        filtered_candidates,
        query_embedding
    )
    
    # 5. Вернуть ровно TOP-10
    return scored_results[:10]
```

### Полная реализация

```python
# backend/search.py
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SearchService:
    def __init__(self, embedding_service, database):
        self.embedding_service = embedding_service
        self.db = database
        
        # Веса для полей (из документации Екатерины)
        self.field_weights = {
            'FAS_arguments': 1.0,
            'violation_summary': 0.8,
            'addescription': 0.6
        }
    
    def search(self, query: str, filters: Dict = None) -> List[Dict]:
        """
        Главная функция поиска
        
        Args:
            query: Поисковый запрос пользователя
            filters: Словарь фильтров {year, region, industry, article}
        
        Returns:
            Список из 10 документов с метаданными
        """
        # Шаг 1: Эмбеддинг запроса
        query_embedding = self.embedding_service.embed_text(
            query,
            task_type="retrieval_query"
        )
        
        # Шаг 2: Поиск TOP-50 по всем документам
        top_50 = self._vector_search_top50(query_embedding)
        
        # Шаг 3: Применить фильтры
        filtered = self._apply_filters(top_50, filters or {})
        
        # Шаг 4: Взвешенная сортировка
        scored = self._calculate_weighted_scores(filtered, query_embedding)
        
        # Шаг 5: Вернуть TOP-10
        return scored[:10]
    
    def _vector_search_top50(self, query_embedding: np.ndarray) -> List[Dict]:
        """
        Векторный поиск TOP-50 кандидатов
        
        Поиск ведется по полю FAS_arguments (самый важный вес 1.0)
        """
        # SQL запрос с pgvector
        query = """
            SELECT 
                id,
                document_url,
                document_date,
                FASdivision,
                defendant_industry,
                legal_provisions,
                FAS_arguments,
                FAS_arguments_embedding,
                violation_summary,
                violation_summary_embedding,
                addescription,
                addescription_embedding,
                (1 - (FAS_arguments_embedding <=> %s::vector)) as similarity
            FROM documents
            ORDER BY FAS_arguments_embedding <=> %s::vector
            LIMIT 50
        """
        
        # Выполнить запрос
        results = self.db.execute(query, (query_embedding, query_embedding))
        return results
    
    def _apply_filters(self, documents: List[Dict], filters: Dict) -> List[Dict]:
        """
        Применить фильтры к списку документов
        
        Логика:
        - Между фильтрами: AND (пересечение)
        - Внутри фильтра: OR (если несколько значений)
        """
        filtered = documents
        
        # Фильтр по году
        if filters.get('year'):
            years = filters['year'] if isinstance(filters['year'], list) else [filters['year']]
            filtered = [
                doc for doc in filtered
                if doc['document_date'].year in years
            ]
        
        # Фильтр по региону
        if filters.get('region'):
            regions = filters['region'] if isinstance(filters['region'], list) else [filters['region']]
            filtered = [
                doc for doc in filtered
                if doc['FASdivision'] in regions
            ]
        
        # Фильтр по отрасли
        if filters.get('industry'):
            industries = filters['industry'] if isinstance(filters['industry'], list) else [filters['industry']]
            filtered = [
                doc for doc in filtered
                if doc['defendant_industry'] in industries
            ]
        
        # Фильтр по статье (содержит)
        if filters.get('article'):
            articles = filters['article'] if isinstance(filters['article'], list) else [filters['article']]
            filtered = [
                doc for doc in filtered
                if any(art in doc['legal_provisions'] for art in articles)
            ]
        
        return filtered
    
    def _calculate_weighted_scores(self, documents: List[Dict], query_embedding: np.ndarray) -> List[Dict]:
        """
        Рассчитать взвешенные оценки релевантности
        
        Формула: S = 1.0*R_FAS + 0.8*R_violation + 0.6*R_ad
        """
        scored_docs = []
        
        for doc in documents:
            # Косинусное сходство по каждому полю
            scores = {}
            for field in ['FAS_arguments', 'violation_summary', 'addescription']:
                field_embedding = doc[f'{field}_embedding']
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    field_embedding.reshape(1, -1)
                )[0][0]
                
                # Нормализация в [0, 1] (опционально, но рекомендуется)
                scores[field] = max(0.0, min(1.0, similarity))
            
            # Взвешенный финальный балл
            final_score = (
                self.field_weights['FAS_arguments'] * scores['FAS_arguments'] +
                self.field_weights['violation_summary'] * scores['violation_summary'] +
                self.field_weights['addescription'] * scores['addescription']
            )
            
            # Добавить балл в документ
            doc['final_score'] = final_score
            doc['field_scores'] = scores
            scored_docs.append(doc)
        
        # Сортировать по убыванию балла
        scored_docs.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_docs
```

---

## ⚖️ ВЗВЕШЕННАЯ РЕЛЕВАНТНОСТЬ

### Математическая модель

```
S = w_V * R_V + w_B * R_B + w_A * R_A

где:
  S - финальный балл релевантности
  R_V - оценка релевантности поля FAS_arguments
  R_B - оценка релевантности поля violation_summary
  R_A - оценка релевантности поля addescription
  w_V = 1.0 (вес FAS_arguments)
  w_B = 0.8 (вес violation_summary)
  w_A = 0.6 (вес addescription)
```

### Почему именно эти веса?

| Поле | Вес | Обоснование |
|------|-----|-------------|
| **FAS_arguments** | 1.0 | Юридические тезисы, цитаты, позиции ФАС - самое важное |
| **violation_summary** | 0.8 | Краткая логика квалификации нарушения - очень важно |
| **addescription** | 0.6 | Фактическое описание рекламы - важно, но менее критично |

### Пример расчета

**Запрос:** "Реклама алкоголя в интернете"

**Документ #1234:**
```python
# Косинусные сходства по полям:
R_FAS = 0.92  # FAS_arguments очень релевантен
R_violation = 0.75  # violation_summary средне релевантен
R_ad = 0.88  # addescription очень релевантен

# Расчет финального балла:
S = 1.0 * 0.92 + 0.8 * 0.75 + 0.6 * 0.88
S = 0.92 + 0.60 + 0.528
S = 2.048
```

**Документ #5678:**
```python
R_FAS = 0.85
R_violation = 0.90
R_ad = 0.70

S = 1.0 * 0.85 + 0.8 * 0.90 + 0.6 * 0.70
S = 0.85 + 0.72 + 0.42
S = 1.99
```

**Результат:** Документ #1234 (S=2.048) будет выше #5678 (S=1.99), потому что у него выше оценка самого важного поля (FAS_arguments).

### Нормализация оценок

**Зачем:** Чтобы длинный текст в поле с низким весом не перебивал короткий текст в поле с высоким весом.

```python
def normalize_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """
    Нормализация оценки в диапазон [0, 1]
    """
    if score < min_score:
        return 0.0
    if score > max_score:
        return 1.0
    return (score - min_score) / (max_score - min_score)

# Применение:
for field in ['FAS_arguments', 'violation_summary', 'addescription']:
    raw_score = cosine_similarity(query_emb, field_emb)
    normalized = normalize_score(raw_score, min_score=-1.0, max_score=1.0)
    # Теперь normalized в диапазоне [0, 1]
```

---

## 🎚️ СИСТЕМА ФИЛЬТРОВ

### 4 типа фильтров

#### 1. Фильтр "Год решения ФАС"

**Поле БД:** `document_date` (timestamp)

**Логика:**
```python
def filter_by_year(documents: List[Dict], years: List[int]) -> List[Dict]:
    """
    Отфильтровать по году решения
    
    Args:
        documents: Список документов
        years: Список годов, например [2023, 2024]
    
    Returns:
        Документы, где document_date.year in years
    """
    if not years:
        return documents
    
    return [
        doc for doc in documents
        if doc['document_date'].year in years
    ]
```

**Пример:**
```python
# Пользователь выбрал: 2023, 2024
filtered = filter_by_year(documents, [2023, 2024])
# Вернутся только документы 2023 и 2024 годов
```

#### 2. Фильтр "Регион (УФАС)"

**Поле БД:** `FASdivision` (text)

**Значения:**
- "Московское УФАС России"
- "УФАС по г. Москве"
- "УФАС по Санкт-Петербургу"
- и т.д.

**Логика:**
```python
def filter_by_region(documents: List[Dict], regions: List[str]) -> List[Dict]:
    """
    Отфильтровать по региону УФАС
    
    Args:
        documents: Список документов
        regions: Список регионов, например ["Москва", "СПб"]
    
    Returns:
        Документы, где FASdivision in regions
    """
    if not regions:
        return documents
    
    # Нормализация названий регионов
    normalized_regions = [normalize_region_name(r) for r in regions]
    
    return [
        doc for doc in documents
        if normalize_region_name(doc['FASdivision']) in normalized_regions
    ]

def normalize_region_name(region: str) -> str:
    """
    Нормализовать название региона
    
    "Московское УФАС России" -> "Москва"
    "УФАС по г. Москве" -> "Москва"
    """
    mapping = {
        "Московское УФАС России": "Москва",
        "УФАС по г. Москве": "Москва",
        "УФАС по Санкт-Петербургу": "Санкт-Петербург",
        # ... добавить остальные регионы
    }
    return mapping.get(region, region)
```

#### 3. Фильтр "Отрасль лица"

**Поле БД:** `defendant_industry` (text)

**Значения:**
- "Финансовые услуги"
- "Строительство"
- "Розничная торговля"
- "Медицина"
- и т.д.

**Логика:**
```python
def filter_by_industry(documents: List[Dict], industries: List[str]) -> List[Dict]:
    """
    Отфильтровать по отрасли нарушителя
    
    Args:
        documents: Список документов
        industries: Список отраслей
    
    Returns:
        Документы, где defendant_industry in industries
    """
    if not industries:
        return documents
    
    return [
        doc for doc in documents
        if doc['defendant_industry'] in industries
    ]
```

#### 4. Фильтр "Статья нормативного акта"

**Поле БД:** `legal_provisions` (text, может содержать несколько статей)

**Значения:**
- "ст. 5"
- "ст. 24"
- "ст. 28"
- "ст. 5 ч. 7"
- и т.д.

**Логика:**
```python
def filter_by_article(documents: List[Dict], articles: List[str]) -> List[Dict]:
    """
    Отфильтровать по статье закона
    
    Args:
        documents: Список документов
        articles: Список статей, например ["ст. 5", "ст. 24"]
    
    Returns:
        Документы, где legal_provisions содержит хотя бы одну из статей
    """
    if not articles:
        return documents
    
    return [
        doc for doc in documents
        if any(article in doc['legal_provisions'] for article in articles)
    ]
```

**Пример:**
```python
# Пользователь выбрал: "ст. 5", "ст. 24"
filtered = filter_by_article(documents, ["ст. 5", "ст. 24"])

# Документ с legal_provisions = "ст. 5, ст. 7" → включен (содержит "ст. 5")
# Документ с legal_provisions = "ст. 24" → включен (содержит "ст. 24")
# Документ с legal_provisions = "ст. 28" → НЕ включен (не содержит ни "ст. 5", ни "ст. 24")
```

### Комбинирование фильтров

**Правило:** Между фильтрами - AND (пересечение), внутри фильтра - OR (объединение)

**Пример:**
```
Запрос: "Реклама алкоголя"
Фильтры:
  Год = [2023, 2024]
  Регион = ["Москва", "СПб"]
  Статья = ["ст. 5"]

Логика:
  (Год = 2023 OR Год = 2024)
  AND
  (Регион = Москва OR Регион = СПб)
  AND
  (Статья содержит "ст. 5")
```

**Код:**
```python
def apply_all_filters(documents: List[Dict], filters: Dict) -> List[Dict]:
    """
    Применить все фильтры последовательно (AND между фильтрами)
    """
    result = documents
    
    # Фильтр 1: Год (OR внутри)
    if filters.get('year'):
        result = filter_by_year(result, filters['year'])
    
    # Фильтр 2: Регион (OR внутри)
    if filters.get('region'):
        result = filter_by_region(result, filters['region'])
    
    # Фильтр 3: Отрасль (OR внутри)
    if filters.get('industry'):
        result = filter_by_industry(result, filters['industry'])
    
    # Фильтр 4: Статья (OR внутри)
    if filters.get('article'):
        result = filter_by_article(result, filters['article'])
    
    return result
```

---

## 📦 ПРИМЕРЫ ПОЛНОГО КОДА

### Полный API endpoint

```python
# backend/api/search.py
from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    year: Optional[List[int]] = None
    region: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    article: Optional[List[str]] = None

class SearchResult(BaseModel):
    document_url: str
    document_date: str
    FASdivision: str
    defendant_industry: str
    legal_provisions: str
    excerpt: str  # Краткий отрывок
    final_score: float
    field_scores: dict

@router.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """
    API endpoint для поиска документов
    
    POST /api/search
    {
        "query": "Реклама алкоголя в интернете",
        "year": [2023, 2024],
        "region": ["Москва"],
        "article": ["ст. 5"]
    }
    
    Returns:
        Список из максимум 10 документов
    """
    # Подготовить фильтры
    filters = {
        'year': request.year,
        'region': request.region,
        'industry': request.industry,
        'article': request.article
    }
    
    # Выполнить поиск
    results = search_service.search(
        query=request.query,
        filters=filters
    )
    
    # Преобразовать в формат API
    return [
        SearchResult(
            document_url=doc['document_url'],
            document_date=doc['document_date'].isoformat(),
            FASdivision=doc['FASdivision'],
            defendant_industry=doc['defendant_industry'],
            legal_provisions=doc['legal_provisions'],
            excerpt=create_excerpt(doc, request.query),
            final_score=doc['final_score'],
            field_scores=doc['field_scores']
        )
        for doc in results
    ]

def create_excerpt(doc: dict, query: str, max_length: int = 200) -> str:
    """
    Создать короткий отрывок из документа
    """
    # Взять самое релевантное поле
    best_field = max(doc['field_scores'], key=doc['field_scores'].get)
    text = doc[best_field]
    
    # Обрезать до max_length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text
```

### Интеграция с фронтендом

```javascript
// frontend/src/api/search.js
export async function searchDocuments(query, filters) {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      year: filters.year || null,
      region: filters.region || null,
      industry: filters.industry || null,
      article: filters.article || null,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Search failed');
  }
  
  return await response.json();
}

// Использование:
const results = await searchDocuments(
  "Реклама алкоголя в интернете",
  {
    year: [2023, 2024],
    region: ["Москва"],
    article: ["ст. 5"]
  }
);

console.log(`Найдено ${results.length} документов`);
results.forEach(doc => {
  console.log(`${doc.document_url}: ${doc.final_score.toFixed(3)}`);
});
```

---

## ✅ ЧЕКЛИСТ РЕАЛИЗАЦИИ

### Критично (обязательно):
- [ ] Удалили sentence-transformers, установили google-generativeai
- [ ] Получили GOOGLE_API_KEY, добавили в .env
- [ ] Везде заменили 384 → 384
- [ ] Обновили схему БД (vector(384))
- [ ] Пересоздали все эмбеддинги через Google Gemini
- [ ] Реализовали новую архитектуру (TOP-50 → фильтры → TOP-10)
- [ ] Добавили взвешенную релевантность (1.0, 0.8, 0.6)
- [ ] Фильтры применяются ПОСЛЕ поиска
- [ ] Всегда возвращаем ровно 10 результатов

### Желательно:
- [ ] Реализовали все 4 фильтра
- [ ] Добавили нормализацию оценок
- [ ] Добавили золотой стандарт (20 тестов)
- [ ] Обновили UX-тексты
- [ ] Добавили логирование поисковых запросов
- [ ] Оптимизировали SQL-запросы с индексами
- [ ] Добавили кэширование частых запросов

### Удалить:
- [ ] Весь код с sentence-transformers
- [ ] Hybrid search (семантика + ключевые слова)
- [ ] Применение фильтров ДО поиска
- [ ] Все упоминания размерности 384

---

## 🚨 ЧАСТЫЕ ОШИБКИ И РЕШЕНИЯ

### Ошибка 1: GOOGLE_API_KEY не найден
```
ValueError: GOOGLE_API_KEY не найден в .env
```

**Решение:**
1. Создать файл `.env` в корне проекта
2. Добавить строку: `GOOGLE_API_KEY=your_actual_key`
3. Убедиться, что load_dotenv() вызывается до импорта модулей

### Ошибка 2: Неправильная размерность
```
AssertionError: Ожидается 384, получено 384
```

**Решение:**
- Найти все места с `384` и заменить на `384`
- Пересоздать файл эмбеддингов (`embeddings.npy`)
- Пересоздать таблицу БД с vector(384)

### Ошибка 3: Пустая выдача после фильтров
```
Найдено 0 результатов
```

**Решение:**
- Проверить, что фильтры применяются к TOP-50, а не ко всем документам
- Проверить логику OR внутри фильтров
- Добавить логирование: сколько документов после каждого фильтра

### Ошибка 4: Медленный поиск
```
Запрос выполняется > 5 секунд
```

**Решение:**
- Добавить индексы на vector columns:
  ```sql
  CREATE INDEX ON documents USING ivfflat (FAS_arguments_embedding vector_cosine_ops);
  ```
- Использовать батчинг для эмбеддингов
- Кэшировать частые запросы

---

**Версия:** 2.1  
**Дата:** 21 января 2026  
**Статус:** ✅ Готово к реализации