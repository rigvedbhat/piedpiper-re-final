import os
import glob

files = glob.glob('src/**/*.py', recursive=True)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('OPENAI_API_KEY', 'GOOGLE_API_KEY')
    new_content = new_content.replace('OPENAI_MODEL', 'GOOGLE_MODEL')
    new_content = new_content.replace('gpt-4o-mini', 'gemini-1.5-pro')
    new_content = new_content.replace('from langchain_openai import ChatOpenAI', 'from langchain_google_genai import ChatGoogleGenerativeAI')
    new_content = new_content.replace('ChatOpenAI(', 'ChatGoogleGenerativeAI(')
    
    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file}")
