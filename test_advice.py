import os, sys
sys.path.append(os.getcwd())
from agent import run_advisor

question = "I'm thinking of buying Apple. Is now a good time, and does anything look off in my recent transactions?"
print('🔎 Running example query:')
print(question)
print('\n---\n')
try:
    result = run_advisor(question, None)
    print('✅ AdvisorIQ response:')
    print(result.get('answer'))
except Exception as e:
    print(f'⚠️ Error while running advisor: {type(e).__name__}: {e}')
