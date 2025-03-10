from django.shortcuts import render, redirect
from .models import *

# Create your views here.
from .langgraph import graph



def upload(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        
        document = Document.objects.create(file=file)
        document.save()
       
        return redirect('home')
        
    return render(request, 'upload.html')


def home(request):
 
    if 'conversation' not in request.session:
        request.session['conversation'] = []

    conversation = request.session['conversation']

    if request.method == 'POST':
        user_msg = request.POST.get('user_msg', '')

        conversation.append({'sender': 'User', 'message': user_msg})

        initial_input = {'messages': [user_msg]}  
        thread_data = {'configurable': {'thread_id': '5'}} 
        
        response = graph.stream(initial_input, thread_data, stream_mode='values')

        ai_response = ""
        
        for event in response:
            message = event['messages'][-1]
            if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                continue
            ai_response = message.content

        if ai_response:
            conversation.append({'sender': 'AI', 'message': ai_response})

        request.session['conversation'] = conversation
        request.session.modified = True  

    return render(request, 'home.html', {'conversation': conversation})