from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag import ask

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query(request: Request):
    try:
        body = await request.json()
        q = body.get("question", "") if isinstance(body, dict) else ""
        k = body.get("top_k", 3) if isinstance(body, dict) else 3
        res = ask(str(q), k)
    except Exception:
        res = {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
    return JSONResponse(res, headers={"Access-Control-Allow-Origin": "*"})
