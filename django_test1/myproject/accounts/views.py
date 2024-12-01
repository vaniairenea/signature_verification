from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1 == password2:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists')
            else:
                user = User.objects.create_user(username=username, password=password1)
                user.save()
                messages.success(request, 'Account created successfully')
                return redirect('login')
        else:
            messages.error(request, 'Passwords do not match')
    return render(request, 'accounts/register.html')

# views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'Login successful')
            return redirect('home')  # Redirect to 'home' after successful login
        else:
            messages.error(request, 'Invalid credentials')  # Show error message on failed login
    return render(request, 'accounts/login.html')

from django.shortcuts import render, redirect
from .forms import ImageUploadForm

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload_success')
    else:
        form = ImageUploadForm()
    return render(request, 'accounts/upload_image.html', {'form': form})

def upload_success(request):
    return render(request, 'accounts/upload_success.html')
from django.shortcuts import render

def home(request):
    return render(request, 'accounts/home.html')
