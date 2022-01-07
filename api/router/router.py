from fastapi import APIRouter, Header, Depends, HTTPException,File,UploadFile
from typing import Optional, List, Dict, Text
from pydantic import BaseModel
from pprint import pprint
from fastapi.responses import StreamingResponse, FileResponse
import os
from dotenv import load_dotenv
from src.db_api.Neo4jDB import Neo4jDB
from src.schema.schema import Relation, Entity
from api.service.ModelService import ModelService
import time


load_dotenv()

model_router = APIRouter(prefix="/api")

model_service = ModelService.from_config(config_path="api/configs/pipeline_config.yaml")

try:
    db = Neo4jDB(
        uri=os.environ.get("NEO4J_URI"),
        user=os.environ.get("NEO4J_URI"),
        password=os.environ.get("NEO4J_URI")
    )
except:
    print(f'Don`t accept NEO4J: {os.environ.get("NEO4J_URI")} '
          f'user: {os.environ.get("NEO4J_URI")}, password: {os.environ.get("NEO4J_URI")}')
    db = None


@model_router.get("/run")
async def run(text: Text):
    output = await model_service.run(text)
    entities = output.get("entities", [])
    relations = output.get("relations", [])
    try :
        db.create_entities(entities)
        db.create_relations(relations)
    except:
        raise HTTPException(status_code=503, detail="Neo4j DB not is available!")
    return output


@model_router.get("/predict")
async def predict(text: Text):
    s_t = time.time()
    output = await model_service.run(text)
    e_t = time.time()
    output["time"] = e_t - s_t
    return output


@model_router.get("/predict-ner")
async def predict_ner(text: Text):
    output = await model_service.run_ner(text)
    return output


@model_router.get("/version")
def get_version():
    return model_service.get_version()
