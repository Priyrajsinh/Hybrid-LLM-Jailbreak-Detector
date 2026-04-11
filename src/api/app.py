from fastapi import FastAPI

app = FastAPI(title="P1 Hybrid Jailbreak Detector")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
