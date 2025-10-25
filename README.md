# ByteData Backend (Flask + Supabase)

## Como rodar localmente
1. Copie `.env.example` para `.env` e preencha as chaves (GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).
2. Crie e ative o ambiente virtual, instale deps e rode:

3. Testes rápidos:
- http://127.0.0.1:8080/health
- GET http://127.0.0.1:8080/filters/get
- POST http://127.0.0.1:8080/filters/save

## Deploy no Render
1. Suba o repo no GitHub.
2. No Render: New → Web Service → conecte o repo.
- Build: `pip install -r requirements.txt`
- Start: `python app.py`
3. Em Environment Variables, adicione:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `GEMINI_API_KEY`
4. Deploy 🚀
