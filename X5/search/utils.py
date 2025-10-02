import torch
import threading
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import os
from django.conf import settings


class NERPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'trained_ner_model')

        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.tag2id = {}
        self.id2tag = {}
        self._lock = threading.Lock()  # Для thread-safe
        self._loaded = False

        self.load_model()

    def load_model(self):
        """Thread-safe загрузка модели"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:  # Double-check locking
                    print("Загрузка модели NER...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
                    self.model.to(self.device)
                    self.model.eval()

                    with open(os.path.join(self.model_path, 'tag_mappings.json'), 'r', encoding='utf-8') as f:
                        mappings = json.load(f)
                        self.tag2id = mappings['tag2id']
                        self.id2tag = {int(k): v for k, v in mappings['id2tag'].items()}

                    self._loaded = True
                    print(f"Модель загружена успешно. Доступные теги: {list(self.tag2id.keys())}")

    def predict(self, query, max_len=64):
        """Thread-safe предсказание"""
        with self._lock:
            encoding = self.tokenizer(
                query,
                max_length=max_len,
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

            entities = []
            current_entity = None
            start_idx = None
            current_tag = None

            for i, (pred_idx, offset) in enumerate(zip(predictions, offset_mapping)):
                if offset[0] == 0 and offset[1] == 0:
                    if current_entity is not None:
                        entities.append({
                            'start': int(start_idx),
                            'end': int(prev_offset[1]),
                            'entity': current_tag,
                            'text': query[int(start_idx):int(prev_offset[1])]
                        })
                        current_entity = None
                    continue

                tag = self.id2tag[pred_idx]

                if tag.startswith('B-'):
                    if current_entity is not None:
                        entities.append({
                            'start': int(start_idx),
                            'end': int(prev_offset[1]),
                            'entity': current_tag,
                            'text': query[int(start_idx):int(prev_offset[1])]
                        })

                    current_entity = []
                    start_idx = offset[0]
                    current_tag = tag

                elif tag.startswith('I-') and current_entity is not None:
                    if current_tag and tag[2:] == current_tag[2:]:
                        pass
                    else:
                        if current_entity is not None:
                            entities.append({
                                'start': int(start_idx),
                                'end': int(prev_offset[1]),
                                'entity': current_tag,
                                'text': query[int(start_idx):int(prev_offset[1])]
                            })
                        current_entity = None

                else:
                    if current_entity is not None:
                        entities.append({
                            'start': int(start_idx),
                            'end': int(prev_offset[1]),
                            'entity': current_tag,
                            'text': query[int(start_idx):int(prev_offset[1])]
                        })
                        current_entity = None

                prev_offset = offset

            if current_entity is not None:
                entities.append({
                    'start': int(start_idx),
                    'end': int(offset[1]),
                    'entity': current_tag,
                    'text': query[int(start_idx):int(offset[1])]
                })

            return entities


ner_predictor = None


def get_ner_predictor():
    global ner_predictor
    if ner_predictor is None:
        ner_predictor = NERPredictor()
    return ner_predictor