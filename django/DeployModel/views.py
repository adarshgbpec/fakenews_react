from pickle import STRING
from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np

from numpy.core import numeric
def home(request):
    return render(request,'home.html')
def result(request):
    L=joblib.load('finalized_model.sav')
    lis=[]
    lis=(request.POST['news'])
    c = np.array([lis],dtype=str)
    L.predict([[c]])
    # print(ans)
    return render(request,"result.html")    