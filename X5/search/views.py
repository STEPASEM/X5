# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import time

from .forms import SearchForm
from .utils import get_ner_predictor

def search(request):
    form = SearchForm()
    results = None

    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['search']
            try:
                predictor = get_ner_predictor()
                entities = predictor.predict(query)

                results = {
                    'query': query,
                    'entities': entities,
                    'formatted_annotation': str([(e['start'], e['end'], e['entity']) for e in entities])
                }
            except Exception as e:
                results = {'error': str(e)}

    return render(request, 'search/search.html', {
        'form': form,
        'results': results
    })


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """
    API endpoint для хакатона (/api/predict)
    """
    start_time = time.time()

    try:
        # Парсим JSON тело запроса с правильной кодировкой
        if request.content_type == 'application/json':
            try:
                # Декодируем с правильной кодировкой
                body_str = request.body.decode('utf-8')
                body = json.loads(body_str)
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                try:
                    body_str = request.body.decode('cp1251')
                    body = json.loads(body_str)
                except:
                    body_str = request.body.decode('latin-1')
                    body = json.loads(body_str)
            text = body.get('text', '').strip()
            request_id = body.get('request_id', '')
        else:
            # Для form-data
            text = request.POST.get('text', '').strip()
            request_id = request.POST.get('request_id', '')

        if not text:
            return JsonResponse({
                'request_id': request_id,
                'text': text,
                'entities': [],
                'processing_time': time.time() - start_time,
                'status': 'error: empty text'
            }, status=400)

        # Используем ваш существующий predictor
        predictor = get_ner_predictor()
        entities = predictor.predict(text)

        processing_time = time.time() - start_time

        return JsonResponse({
            'request_id': request_id,
            'text': text,
            'entities': entities,
            'processing_time': round(processing_time, 3),
            'status': 'success'
        })

    except json.JSONDecodeError as e:
        return JsonResponse({
            'request_id': '',
            'text': '',
            'entities': [],
            'processing_time': time.time() - start_time,
            'status': f'error: invalid JSON - {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'request_id': '',
            'text': '',
            'entities': [],
            'processing_time': time.time() - start_time,
            'status': f'error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint для мониторинга
    """
    try:
        predictor = get_ner_predictor()
        model_loaded = predictor is not None
    except:
        model_loaded = False

    return JsonResponse({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': time.time(),
        'service': 'NER Prediction API'
    })