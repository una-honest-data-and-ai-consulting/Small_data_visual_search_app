###TO DO

from fastapi import FastAPI
from routers.router import model_router

app = FastAPI()
app.include_router(model_router.router, prefix='/model')  

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Siamese Mask R-CNN is all ready to go'