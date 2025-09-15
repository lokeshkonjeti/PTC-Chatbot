

import google.generativeai as genai

genai.configure(api_key="AIzaSyAUW6HIVH0V0H18IMuCaCrA3VM3qEFOnlM")

model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("Hello, who won the 2022 World Cup?")
print(response.text)