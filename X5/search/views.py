from django.shortcuts import render

from .forms import SearchForm


def search(request):
    form = SearchForm(request.GET or None)
    context = {'form': form}
    return render(request, 'search/search.html', context)
