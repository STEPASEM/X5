import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import ast
import numpy as np


class SearchQueryDataset(Dataset):
    """Датасет для поисковых запросов с NER разметкой"""

    def __init__(self, queries, annotations, tokenizer, max_len, tag2id, is_test=False):
        self.queries = queries
        self.annotations = annotations if not is_test else [None] * len(queries)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id
        self.is_test = is_test

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        query = str(self.queries[item])

        # Создаем символьные метки для каждого символа в запросе
        char_labels = ['O'] * len(query)

        if not self.is_test:
            annotation = self.annotations[item]
            for annot in annotation:
                if len(annot) == 3:  # Проверяем формат (start, end, tag)
                    start, end, tag = annot
                    # Обеспечиваем корректные индексы
                    start = max(0, min(start, len(query) - 1))
                    end = max(start + 1, min(end, len(query)))

                    char_labels[start] = tag
                    # Помечаем внутренние части сущности как I-*
                    for i in range(start + 1, end):
                        inner_tag = tag.replace('B-', 'I-')
                        char_labels[i] = inner_tag

        # Токенизация с учетом выравнивания
        encoding = self.tokenizer(
            query,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        # Сопоставляем символьные метки с токенами
        labels = []
        offset_mapping = encoding['offset_mapping'].squeeze()

        for i, offset in enumerate(offset_mapping):
            if offset[0] == 0 and offset[1] == 0:  # Специальные токены
                labels.append(-100)
            else:
                # Берем метку первого символа токена
                char_index = offset[0]
                if char_index < len(char_labels):
                    label = char_labels[char_index]
                    labels.append(self.tag2id.get(label, self.tag2id['O']))
                else:
                    labels.append(self.tag2id['O'])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class NERModel:
    """Класс для работы с NER моделью"""

    def __init__(self, model_name='DeepPavlov/rubert-base-cased', max_len=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.tag2id = {}
        self.id2tag = {}

    def prepare_data(self, train_csv_path):
        """Подготовка данных и словаря тегов"""
        print("Загрузка данных...")
        # Читаем CSV с правильным разделителем
        train_df = pd.read_csv(train_csv_path, sep=';')

        # Собираем уникальные теги
        unique_tags = set()
        for annotation in train_df['annotation']:
            try:
                list_of_annots = ast.literal_eval(annotation)
                for annot in list_of_annots:
                    if len(annot) == 3:
                        unique_tags.add(annot[2])
            except:
                continue

        unique_tags.add('O')
        self.tag2id = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
        self.id2tag = {idx: tag for tag, idx in self.tag2id.items()}

        print(f"Найдено тегов: {len(self.tag2id)}")
        print("Теги:", self.tag2id)

        return train_df

    def create_data_loaders(self, train_df, batch_size=16, val_size=0.2):
        """Создание DataLoader для обучения и валидации"""
        # Преобразуем аннотации
        queries = train_df['sample'].values  # Изменено с 'search_query' на 'sample'
        annotations = []
        for ann in train_df['annotation']:
            try:
                annotations.append(ast.literal_eval(ann))
            except:
                annotations.append([])

        # Разделяем на train/val
        train_queries, val_queries, train_annots, val_annots = train_test_split(
            queries, annotations, test_size=val_size, random_state=42
        )

        # Инициализируем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Создаем датасеты
        train_dataset = SearchQueryDataset(
            train_queries, train_annots, self.tokenizer, self.max_len, self.tag2id
        )
        val_dataset = SearchQueryDataset(
            val_queries, val_annots, self.tokenizer, self.max_len, self.tag2id
        )

        # Создаем DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def initialize_model(self):
        """Инициализация модели"""
        print(f"Инициализация модели {self.model_name}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.tag2id),
            id2label=self.id2tag,
            label2id=self.tag2id
        )
        self.model.to(self.device)

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Обучение модели"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        print("Начало обучения...")
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            avg_train_loss = total_loss / len(train_loader)

            # Валидация
            val_loss, val_f1 = self.validate(val_loader)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val F1: {val_f1:.4f}')
            print('-' * 50)

    def validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

                # Предсказания для метрик
                predictions = torch.argmax(outputs.logits, dim=2)

                # Собираем метки, игнорируя -100
                for i in range(len(labels)):
                    mask = labels[i] != -100
                    true_labels = labels[i][mask].cpu().numpy()
                    pred_labels = predictions[i][mask].cpu().numpy()

                    all_true_labels.extend(true_labels)
                    all_pred_labels.extend(pred_labels)

        avg_loss = total_loss / len(val_loader)

        # Вычисляем F1-score
        if len(all_true_labels) > 0:
            f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
        else:
            f1 = 0.0

        return avg_loss, f1

    def predict_single(self, query):
        """Предсказание для одного запроса"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не инициализирована. Сначала вызовите initialize_model()")

        self.model.eval()

        encoding = self.tokenizer(
            query,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'].squeeze().cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()

        # Восстанавливаем сущности из предсказаний
        entities = []
        current_entity = None
        start_idx = None
        current_tag = None

        for i, (pred_idx, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:  # Специальные токены
                continue

            tag = self.id2tag[pred_idx]

            if tag.startswith('B-'):
                # Сохраняем предыдущую сущность
                if current_entity is not None:
                    entities.append({
                        'start': start_idx,
                        'end': prev_offset[1],
                        'entity': current_tag,
                        'text': query[start_idx:prev_offset[1]]
                    })

                # Начинаем новую сущность
                current_entity = []
                start_idx = offset[0]
                current_tag = tag

            elif tag.startswith('I-') and current_entity is not None:
                # Проверяем, что это продолжение той же сущности
                if current_tag and tag[2:] == current_tag[2:]:
                    continue  # Просто продолжаем
                else:
                    # Разные сущности - сохраняем текущую и начинаем новую
                    if current_entity is not None:
                        entities.append({
                            'start': start_idx,
                            'end': prev_offset[1],
                            'entity': current_tag,
                            'text': query[start_idx:prev_offset[1]]
                        })
                    current_entity = None

            else:  # 'O' или другая сущность
                if current_entity is not None:
                    entities.append({
                        'start': start_idx,
                        'end': prev_offset[1],
                        'entity': current_tag,
                        'text': query[start_idx:prev_offset[1]]
                    })
                    current_entity = None

            prev_offset = offset

        # Добавляем последнюю сущность
        if current_entity is not None:
            entities.append({
                'start': start_idx,
                'end': offset[1],
                'entity': current_tag,
                'text': query[start_idx:offset[1]]
            })

        return entities

    def predict_csv(self, test_csv_path, output_csv_path=None):
        """Предсказание для CSV файла"""
        print(f"Загрузка тестовых данных из {test_csv_path}...")
        # Читаем CSV с правильным разделителем
        test_df = pd.read_csv(test_csv_path, sep=';')

        results = []
        print("Выполнение предсказаний...")

        for idx, row in test_df.iterrows():
            query = str(row['sample'])  # Изменено с 'search_query' на 'sample'
            entities = self.predict_single(query)

            # Форматируем в требуемый формат BIO
            bio_annotation = []
            for entity in entities:
                bio_annotation.append((
                    entity['start'],
                    entity['end'],
                    entity['entity']
                ))

            # Сохраняем результат с правильными названиями колонок
            result = {
                'sample': query,
                'annotation': str(bio_annotation)  # Сохраняем как строку
            }

            # Если есть ID, сохраняем его
            if 'id' in row:
                result['id'] = row['id']

            results.append(result)

            # Вывод в консоль
            print(f"\nЗапрос {idx + 1}: '{query}'")
            print("Сущности:")
            for entity in entities:
                print(f"  {entity['entity']}: '{entity['text']}' (позиции {entity['start']}-{entity['end']})")
            print("-" * 60)

        # Сохранение результатов
        if output_csv_path:
            output_df = pd.DataFrame(results)
            # Сохраняем с тем же разделителем, что и исходный файл
            output_df.to_csv(output_csv_path, index=False, sep=';')
            print(f"\nРезультаты сохранены в {output_csv_path}")

        return results

    def save_model(self, path):
        """Сохранение модели"""
        if self.model and self.tokenizer:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            # Сохраняем словари тегов отдельно
            import json
            with open(f'{path}/tag_mappings.json', 'w', encoding='utf-8') as f:
                json.dump({'tag2id': self.tag2id, 'id2tag': self.id2tag}, f, ensure_ascii=False)

            print(f"Модель сохранена в {path}")

    def load_model(self, path):
        """Загрузка модели"""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.model.to(self.device)

        # Загружаем словари тегов
        import json
        try:
            with open(f'{path}/tag_mappings.json', 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.tag2id = mappings['tag2id']
                self.id2tag = {int(k): v for k, v in mappings['id2tag'].items()}
        except:
            # Если файла нет, используем конфигурацию модели
            self.id2tag = self.model.config.id2label
            self.tag2id = self.model.config.label2id

        print(f"Модель загружена из {path}")


def main():
    """Основная функция"""
    # Параметры
    MODEL_NAME = 'distilbert-base-multilingual-cased'  # Используем предобученную модель
    TRAIN_CSV = 'train.csv'  # Используем ваш файл как обучающие данные
    TEST_CSV = 'submission.csv'  # Тестируем на другом файле
    MAX_LEN = 64
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Инициализация модели
    ner_model = NERModel(model_name=MODEL_NAME, max_len=MAX_LEN)

    try:
        # Подготовка данных
        train_df = ner_model.prepare_data(TRAIN_CSV)

        # Создание DataLoader
        train_loader, val_loader = ner_model.create_data_loaders(
            train_df, batch_size=BATCH_SIZE
        )

        # Инициализация и обучение модели
        ner_model.initialize_model()
        ner_model.train(train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

        # Сохранение модели
        ner_model.save_model('trained_ner_model')

    except FileNotFoundError:
        print(f"Файл {TRAIN_CSV} не найден. Загружаем предобученную модель...")
        # Если нет обучающих данных, пытаемся загрузить существующую модель
        try:
            ner_model.load_model('trained_ner_model')
        except:
            print("Предобученная модель не найдена. Инициализируем новую модель...")
            ner_model.initialize_model()

    # Предсказание для тестовых данных
    try:
        ner_model.predict_csv(TEST_CSV, 'submission_predictions.csv')
    except FileNotFoundError:
        print(f"Файл {TEST_CSV} не найден. Запускаем демо-предсказания...")

        # Демо-предсказания на примерах
        demo_queries = [
            "форма для выпечки",
            "фарш свиной",
            "сок ананасовый без сахара",
            "еринги",
            "молооко"
        ]

        print("\nДемо-предсказания:")
        print("=" * 60)

        for i, query in enumerate(demo_queries):
            entities = ner_model.predict_single(query)
            print(f"\nЗапрос {i + 1}: '{query}'")
            print("Обнаруженные сущности:")

            if entities:
                for entity in entities:
                    print(f"  {entity['entity']}: '{entity['text']}' (позиции {entity['start']}-{entity['end']})")
            else:
                print("  Сущности не обнаружены")
            print("-" * 40)


if __name__ == "__main__":
    main()