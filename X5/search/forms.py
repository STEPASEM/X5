from django import forms

from .models import Search

class SearchForm(forms.ModelForm):

    class Meta:
        model = Search
        fields = '__all__'
        widgets = {
            'search': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Search'}),
        }