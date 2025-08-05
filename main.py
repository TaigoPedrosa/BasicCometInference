#FastAPI HTTP Server and CORS
from ray import serve
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Configs
from os import getenv
MODEL = getenv('MODEL', "Unbabel/wmt22-cometkiwi-da")
NUM_GPUS = int(getenv("NUM_GPUS", 1))
NUM_CPUS = int(getenv("NUM_CPUS", 1))

#Set up logging
import logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)

#Audio processing
import asyncio
from comet import download_model, load_from_checkpoint
from torch import set_num_threads

@serve.deployment(
    ray_actor_options={
        "num_cpus": NUM_CPUS, 
        "num_gpus": 1
    },
    num_replicas=NUM_GPUS
)
@serve.ingress(fastapi_app)
class Main:
    def __init__(self):
        set_num_threads(NUM_CPUS)
                
        logging.warning(f"MODEL[Loading][{MODEL}]")
        self.model = load_from_checkpoint(download_model(MODEL))
        logging.warning(f"MODEL[Loaded][{MODEL}]")
            
    async def process(self, items:list[dict[str,str]]) -> dict[str,str]:
        return self.model.predict(items, batch_size=100, gpus=1, num_workers=NUM_CPUS)

    @fastapi_app.post("/batch")
    async def batch(self, request: Request) -> JSONResponse:
        try:
            items = await request.json()
            await asyncio.sleep(0) #Yield control to the event loop
            
            result = await self.process(items)
            await asyncio.sleep(0) #Yield control to the event loop
            
            return result
            
        except Exception as e:
            logging.error(e)
            return JSONResponse({"message":"Não foi possível processar.", "error": str(e)}, status_code=500)

app = Main.bind()

if __name__ == "__main__":
    serve.api.run(app)