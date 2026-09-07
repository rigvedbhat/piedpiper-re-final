import os
import glob

files = glob.glob('src/**/*.py', recursive=True)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('GOOGLE_API_KEY', 'OPENROUTER_API_KEY')
    new_content = new_content.replace('GOOGLE_MODEL', 'OPENROUTER_MODEL')
    new_content = new_content.replace('gemini-flash-latest', 'google/gemini-1.5-flash')
    new_content = new_content.replace('from langchain_google_genai import ChatGoogleGenerativeAI', 'from langchain_openai import ChatOpenAI')
    new_content = new_content.replace('ChatGoogleGenerativeAI(', 'ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), ')
    
    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file}")
