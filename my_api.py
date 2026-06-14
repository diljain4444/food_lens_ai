from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from new_backend import extract_information,workflow,recipe_chat
from typing import Any
@app.get("/health")
def health():
    return{"status":"ok"}

@app.post("/extract_ingredients")
async def extract_ingredients(image:UploadFile=(File(...))):
    if image.content_type not in ["image/jpeg","image/jpg","image/png"]:
        raise HTTPException(status_code=400,detail="the image should be only in jprg,png,jpg format")
    
    image_bytes=await image.read()
    pil_images=Image.open(BytesIO(image_bytes))
    
    try:
        ingredients=extract_information(pil_images)
    except Exception:
        raise HTTPException(status_code=503,detail="the gemini model is currently not available")
        
    
    return {"detected_ingredients":ingredients}
from typing import List
class Analyse_final_list(BaseModel):
    final_ingredients:List[str]

# out new endpoint for our workflowww
@app.post("/main_endpoint")
async def major_extraction(ingredients:Analyse_final_list):
    if not ingredients.final_ingredients:
        raise HTTPException(status_code=400, detail="Ingredient list cannot be empty")
    
    try:
        result=workflow.invoke({"initial_info":ingredients.final_ingredients})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
    alllll = []
    for recipe, ingr, nutr in zip(
        result["recipe_list"],
        result["ingredients_list"],
        result["nutrition_list"]       # ← now a plain dict
    ):
        item = {}
        item["dish_name"]    = recipe.dish_name
        item["full_recipe"]  = recipe.full_recipe

        # key_ingredients → list of Pydantic objects → convert to dict
        item["key_ingredients"] = [
            {
                "ingredient_name": k.ingredients_name,
                "quantity": k.quantity
            }
            for k in ingr.dishes[0].key_ingredients
        ]

        # nutrition → already plain dict from nutritional_value_agent
        item["ingredients_info"]      = nutr["ingredients_info"]
        item["overall_calories"]      = nutr["overall_calories"]
        item["overall_protein"]       = nutr["overall_protein"]
        item["overall_fats"]          = nutr["overall_fats"]
        item["overall_carbohydrates"] = nutr["overall_carbohydrates"]
        item["overall_vitamins"]      = nutr["overall_vitamins"]

        alllll.append(item)

    return {"final_results": alllll}

class RecipeChatRequest(BaseModel):
    dish_info: Any
    user_query: str

@app.post("/recipe_chat")
async def recipe_chat_endpoint(request: RecipeChatRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        response = recipe_chat(request.dish_info, request.user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    return {"response": response}

# ── Serve frontend ────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse
import os
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

