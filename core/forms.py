from django import forms

class Info(forms.Form):
    Name = forms.CharField(label='Your name', max_length=100)
    Symptom = forms.CharField(label='Mention Symptom', max_length=100)
    No_Days = forms.CharField(label='Enter Number of Days having Symptom', max_length=100)

class symp(forms.Form):
    Exist = forms.CharField(label='yes', max_length=100)
    Qst = forms.CharField(label='qst', max_length=100)
    Noq = forms.CharField(label='no', max_length=100)

class nod(forms.Form):
    Existes  = forms.CharField(label='Number Of Days', max_length=100)
