from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from core.forms import Info, symp, nod
from core.chat_bot import *

@csrf_exempt
def info(request):
    form = Info()
    if request.method == 'POST':
        form = Info(request.POST)
        form2 = nod(request.POST)
        form1 = symp(request.POST)
        if form2.is_valid():
            # print("Enteredd5555555555555555555555555")
            request.session['nod'] = form2.cleaned_data['Existes']
            ans=symptom_generator(request.session['symptom'], request.session['no_days'], int(request.session['nod']))
            form1 = symp()
            noq = len(ans)
            for i in ans:
                request.session[i]="ch"
            qst = ans[0]
            return render(request, 'qst.html', {'form': form1, 'no_que': noq, 'qst': qst})
        if form.is_valid():
            request.session['name'] = form.cleaned_data['Name']
            request.session['symptom'] = form.cleaned_data['Symptom']
            request.session['no_days'] = form.cleaned_data['No_Days']
            msg = initial_executer(request.session['symptom'], request.session['no_days'])
            form2 = nod()
            print(msg)
            print("Enteredd5555555555555555555555555")
            return render(request, 'severe.html', {'form': form2, 'msg': msg})


        if form1.is_valid():
            noq = int(form1.cleaned_data['Noq'])
            qst = form1.cleaned_data['Qst']
            value = form1.cleaned_data['Exist']
            request.session[qst] = value
            if noq != 0:
                noq=noq-1
                for key, value in request.session.items():
                    if request.session[key] == "ch":
                        qst = key
                        break

                return render(request, 'qst.html', {'form': form1, 'no_que': noq, 'qst': qst})
            else:
                listed=[]
                for key, value in request.session.items():
                    if request.session[key] == "yes":
                        listed.append(key)
                ans=result_generator(request.session['symptom'], request.session['no_days'], int(request.session['nod']), listed)
                mes=measure_generator(request.session['symptom'], request.session['no_days'], int(request.session['nod']), listed)

                return render(request, 'final.html', {'res': ans, 'measure':mes})
    return render(request, 'login.html', {'form': form})
