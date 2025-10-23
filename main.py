from fastapi import FastAPI

app = FastAPI()

from omnitool.omniparserserver.omniparserserver import router as omniparserserver_router


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


app.include_router(omniparserserver_router, tags=["omniparserserver"])
