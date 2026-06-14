# multi_back.py
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,List
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
import os
import json
import io
load_dotenv()
import re
api_key = os.getenv("GEMINI_API_KEY")
from google import genai
import pandas as pd
df = pd.read_csv("final.csv")

df["vitamins"] = df["vitamins"].fillna("")
df["Minerals"] = df["Minerals"].fillna("")

# Step 2: Fix numeric columns SECOND

numeric_cols = ["calories", "protein", "fats", "carbohydrates", "fiber"]
df[numeric_cols] = df[numeric_cols].fillna(0)
client = genai.Client(api_key=api_key)
from langchain_groq import ChatGroq
model_text = ChatGroq(
    # model="meta-llama/llama-4-scout-17b-16e-instruct",
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)
model_parse=ChatGroq(
model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)



def extract_information(image):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image,
            """
            You are an ingredient detection system.

    Task:
    Identify ONLY the food ingredients that are clearly visible in the image.

    Rules:

    1. Do NOT guess ingredients that are not visible.
    2. Do NOT infer ingredients based on common recipes.
    3. Do NOT add spices, oil, salt, or seasonings unless they are clearly visible.
    4. If uncertain, do not include the ingredient.
    5. Include chopped, sliced, or partially visible ingredients if they can be identified 
    6. Return ONLY valid JSON.
    7. No markdown, no explanations, no extra text.

    Analyze the image now.

            """
        ]
    )
    clean_text = response.text.replace("```json", "").replace("```", "").strip()
    data = json.loads(clean_text)
    # ingredients = data["ingredients"]
    # return ingredients
    if isinstance(data, dict):
        return data.get("ingredients", [])
    elif isinstance(data, list):
        return data
    return []


from langchain_core.prompts import PromptTemplate
from typing import TypedDict,List
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

from typing import Any
class state(TypedDict):
    initial_info:Any
    recipe_list:Any
    ingredients_list:Any
    nutrition_list:Any

# ============================================
class recipe_schema(BaseModel):
    dish_name:str=Field(description="name of the recipe")
    full_recipe:List[str]=Field(description="detailed recipe step by step for the particular dish")
class multi_recipies(BaseModel):
    multi:List[recipe_schema]=Field(description="multiple dishes")
    
recipe_parser = PydanticOutputParser(pydantic_object=recipe_schema)
multi_dishes_parser=PydanticOutputParser(pydantic_object=multi_recipies)

def recipe_agent(state):
    data=state["initial_info"]
    ingredients=data
    prompt = PromptTemplate(
        template="""
    You are an expert chef.

    Your task is to create the best possible dish using the ingredients provided by the user.

    Ingredients:
    {ingredients}

    Instructions:
    note that you  have to provide the quantity used for all ingredients 
    1. Analyze the available ingredients carefully.
    2. Create 3-4 dish that can realistically be prepared using these ingredients.
    3. You may use common pantry items such as salt, water, oil, and basic spices if required, u can add it without any problem.
    4. Do not introduce major ingredients that are not present in the provided list.
    5. Generate a clear and detailed step-by-step recipe.
    
    6. Each step should be concise and actionable.
    7. Return your response ONLY in the specified JSON format.
    8. note that it is not necessary to use each ingredient in the particular recipe you may and you may not be but note that i want the bestest recipe

    {format_instructions}
    """,
        input_variables=["ingredients"],
        partial_variables={
            "format_instructions": multi_dishes_parser.get_format_instructions()
        }
    )
    chain=prompt|model_text|multi_dishes_parser
    result=chain.invoke({"ingredients":ingredients})
    return {"recipe_list":result.multi}


class ingredients(BaseModel):
    ingredients_name:str=Field(description="name of the ingredient")
    quantity:str=Field(description="the quantity of the ingredient")

class dish_ingredients(BaseModel):
    dish_name: str = Field(description="name of the dish")
    key_ingredients: List[ingredients] = Field(description="list of major ingredients with their quantity used in this dish ")

class all_dishes_ingredients(BaseModel):
    dishes: List[dish_ingredients] = Field(description="key ingredients for each dish")

ingredient_extractor_parser = PydanticOutputParser(pydantic_object=all_dishes_ingredients)

def ingredients_find_agent(state):
    all_recipes=state["recipe_list"]
    all_items=[]
    for item in all_recipes:
      dish_name=(item.dish_name)
      full_recipe=" ".join(item.full_recipe)
      recipe=(dish_name+full_recipe)
      
      prompt = PromptTemplate(
        template="""

You are a professional chef and ingredient analyst.

You will be given a recipe.

Your task is to identify all major food ingredients used in the dish and estimate their quantities.

Recipe:
{recipe}

Rules:

1. Extract ONLY real food ingredients.
2. Do NOT include utensils, cookware, cooking actions, or cooking methods.
3. Do NOT include water.
4. Include oils, ghee, butter, salt, and spices only if they are clearly used in the recipe.
5. If an ingredient quantity is explicitly mentioned, use that quantity exactly.
6. If quantity is not mentioned, estimate a reasonable quantity based on a typical serving size.

NAMING RULES — VERY IMPORTANT:
- Use simple, common English names only
- Do NOT use scientific names or regional names
- Use these exact mappings:
  * Cilantro → coriander leaves
  * Spring onion / Scallion → spring onion
  * Bell pepper → green bell pepper OR red bell pepper OR yellow bell pepper
  * Chili / Chile → green chilli OR red chilli
  * Eggplant → eggplant
  * Zucchini → zucchini
  * Capsicum → green bell pepper
  * Coriander → coriander leaves
  * Methi → fenugreek leaves
  * Pudina → mint leaves
  * Jeera → cumin seeds
  * Haldi → turmeric powder
  * Atta → whole wheat flour
  * Maida → refined flour
  * Besan → chickpea flour
  * Rajma → kidney beans
  * Chana → chickpeas
  * Moong → split green gram
  * Urad → split black gram
  * Toor → split pigeon pea
  * Cooking oil / Vegetable oil / Any neutral oil → oil

IMPORTANT:

- Quantity MUST contain units.
- Use units such as:
  - g
  - kg
  - tbsp
  - tsp
  - ml

- Keep ingredient names clean and standardized.

Good examples:

Paneer -> "100g"
Onion -> "100g"
Tomato -> "200g"
Oil -> "1 tbsp (10g)"
Cumin Seeds -> "1 tsp (5g)"

Return ONLY valid JSON.

Example:

{{
  "dishes": [
    {{
      "dish_name": "Paneer Bhurji",
      "key_ingredients": [
        {{
          "ingredients_name": "Paneer",
          "quantity": "200g"
        }},
        {{
          "ingredients_name": "Onion",
          "quantity": "100g"
        }},
        {{
          "ingredients_name": "Tomato",
          "quantity": "150g"
        }},
        {{
          "ingredients_name": "Oil",
          "quantity": "1 tbsp (10g)"
        }}
      ]
    }}
  ]
}}

No markdown.
No explanation.
No extra text.

{format_instructions}
""",input_variables=["recipe"],partial_variables={"format_instructions": ingredient_extractor_parser.get_format_instructions()}
    )

      chain = prompt | model_text | ingredient_extractor_parser
      result = chain.invoke({"recipe":recipe})
      all_items.append(result)
    return  {"ingredients_list":all_items}
    


def nutrition_calc(product):
    product_clean = product.lower().strip()
    
    # Step 1: Exact match
    mask = df["Ingredient_name"] == product_clean
    if mask.any():
        return df.loc[mask].to_dict(orient="records")[0]
    
    # Step 2: CSV name is contained IN the product name
    # "spinach leaves" contains "spinach" → match
    # "garlic clove"   contains "garlic"  → match
    # "green onions"   contains "onion"   → match
    for db_name in df["Ingredient_name"]:
        if db_name in product_clean:
            mask = df["Ingredient_name"] == db_name
            return df.loc[mask].to_dict(orient="records")[0]
    
    # Step 3: Product name is contained IN the CSV name
    # "egg" is in "eggs" → match
    for db_name in df["Ingredient_name"]:
        if product_clean in db_name:
            mask = df["Ingredient_name"] == db_name
            return df.loc[mask].to_dict(orient="records")[0]

    # Step 4: Word level match
    # "red bell pepper" → check if any word matches "red pepper" or "bell pepper"
    product_words = set(product_clean.split())
    best_match = None
    best_score = 0
    for db_name in df["Ingredient_name"]:
        db_words = set(db_name.split())
        common = len(product_words & db_words)
        if common > best_score and common >= 1:
            best_score = common
            best_match = db_name
    
    if best_match:
        mask = df["Ingredient_name"] == best_match
        return df.loc[mask].to_dict(orient="records")[0]

    # Step 5: Not found
    return {
        "Ingredient_name": product,
        "Category": "Unknown",
        "calories": 0,
        "protein": 0,
        "fats": 0,
        "carbohydrates": 0,
        "fiber": 0,
        "vitamins": "",
        "Minerals": "",
        "Notes / Aliases": ""
    }


UNIT_TO_GRAMS = {
    "tbsp": 15, "tsp": 5, "cup": 240,
    "kg": 1000, "l": 1000, "oz": 28.35,
    "piece": 100, "medium": 100, "large": 150, "small": 70,
}

def extract_quantity(quantity_text):
    quantity_text = quantity_text.lower().strip()

    # Priority 1: brackets like "1 tbsp (10g)" or "(200ml)"
    match = re.search(r"\((\d+(?:\.\d+)?)\s*(g|ml)\)", quantity_text)
    if match:
        return float(match.group(1)), match.group(2)

    # Priority 2: plain grams or ml like "200g" or "200 grams"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(grams|gram|g|ml|milliliter)", quantity_text)
    if match:
        return float(match.group(1)), "g"

    # Priority 3: convert units like "2 tbsp" or "1 tsp"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(tbsp|tsp|cup|kg|oz|l)\b", quantity_text)
    if match:
        qty = float(match.group(1))
        unit = match.group(2)
        grams = qty * UNIT_TO_GRAMS.get(unit, 100)
        return grams, "g"

    # Priority 4: count with size like "2 medium" or "1 large"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(medium|large|small|piece|pieces)\b", quantity_text)
    if match:
        qty = float(match.group(1))
        size = match.group(2)
        grams = qty * UNIT_TO_GRAMS.get(size, 100)
        return grams, "g"

    # Priority 5: bare number — assume grams
    match = re.search(r"(\d+(?:\.\d+)?)", quantity_text)
    if match:
        return float(match.group(1)), "g"

    # Fallback
    return 100, "g"   # ✅ return 100g instead of 0 so nutrition isn't wiped out

def nutritional_value_agent(state):
    all_ingre=state["ingredients_list"]
    final_data=[]
    for item in all_ingre:
        main_info={}
        protein=0
        calories=0
        fats=0
        carbohydrates=0
        vitaminn=set()
        main_info["dish_name"]=item.dishes[0].dish_name
        temp=(item.dishes[0].key_ingredients)
        alll=[]
        for i in temp:
            ins={}
            ins["ingredient_name"]=(i.ingredients_name)
            ins["quantity"]=(i.quantity)
            quantityyy=i.quantity
            quantity, unit = extract_quantity(quantityyy)
            # function calling
            result=nutrition_calc(i.ingredients_name)
            calc_protein= (float(result["protein"]) * quantity) / 100
            calc_fats = (float(result["fats"]) * quantity) / 100
            calc_calories = (float(result["calories"]) * quantity) / 100
            calc_carbohydrates= (float(result["carbohydrates"]) * quantity) / 100
            calc_vitamins=result["vitamins"]
            ins["protein"] = calc_protein
            ins["fats"] = calc_fats
            ins["calories"] = calc_calories
            ins["carbohydrates"] = calc_carbohydrates
            ins["vitamins"]=calc_vitamins
            
            protein+=calc_protein
            calories+=calc_calories
            fats+=calc_fats
            carbohydrates+=calc_carbohydrates
            if isinstance(calc_vitamins, str):
                for v in calc_vitamins.split(","):
                    v = v.strip()
                    if v:
                        vitaminn.add(v)
            elif isinstance(calc_vitamins, list):
                for v in calc_vitamins:
                    v = str(v).strip()
                    if v:
                        vitaminn.add(v)
                
            alll.append(ins)
        vitamins = list(vitaminn)
        main_info["ingredients_info"]=alll
        
        main_info["overall_protein"]       = round(protein, 2)
        main_info["overall_calories"]      = round(calories, 2)
        main_info["overall_fats"]          = round(fats, 2)
        main_info["overall_carbohydrates"] = round(carbohydrates, 2)
        main_info["overall_vitamins"]=vitamins
        final_data.append(main_info)
    return {"nutrition_list":final_data}


def recipe_chat(dish_info,user_query):
    prompt = PromptTemplate(
        template="""You are an expert chef assistant.

The user is asking about or wants to modify the following recipe:

{recipe_context}

User Request:
{user_query}

Instructions:
1. If the user wants to modify the recipe — provide the updated recipe steps clearly
2. If the user wants to substitute an ingredient — suggest the best substitute with quantity
3. If the user wants more details about a step — explain it clearly
4. If the user asks about quantity changes — adjust and confirm all affected steps
5. If the user asks a general question about the recipe — answer directly
6. Keep your response concise and practical
7. Do not make up ingredients that weren't in the original recipe unless user explicitly asks to add something new

Respond in plain text. Be direct and helpful.
""",
        input_variables=["recipe_context", "user_query"]
    )
    
    chain = prompt | model_text
    result = chain.invoke({
        "recipe_context": dish_info,
        "user_query": user_query
    })
    return result.content


from langgraph.graph import StateGraph,START,END
graph=StateGraph(state)

graph = StateGraph(state)

graph.add_node("recipe_agent", recipe_agent)
graph.add_node("ingredients_find_agent", ingredients_find_agent)
graph.add_node("nutritional_breakdown_agent", nutritional_value_agent)


graph.add_edge(START,"recipe_agent")
graph.add_edge("recipe_agent", "ingredients_find_agent")
graph.add_edge("ingredients_find_agent", "nutritional_breakdown_agent")
graph.add_edge("nutritional_breakdown_agent",END)

workflow=graph.compile()
    
        
        