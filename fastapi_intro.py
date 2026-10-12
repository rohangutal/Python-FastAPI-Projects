from fastapi import FastAPI

# creating app
app = FastAPI()

# defining api endpoint
@app.get("/")
def first_example():
    return {"message": "Hello, FastAPI!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_intro:app", host="127.0.0.1", port=8000, reload=True)
