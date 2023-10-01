# client.py
import requests

filename = "requirements.txt"
files = {'my_file': (filename, open(filename, 'rb'))}
# json = {'first': "Hello", 'second': "World"}

response = requests.post(
    'http://127.0.0.1:8000/object_detection/',
    files=files
    # data={'first': "Hello", 'second': "World"}
)
print(response.json())