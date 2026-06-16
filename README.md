# 🍽️ FoodLens AI

**Turn your fridge into gourmet.** Upload a photo of your ingredients, let AI identify what's in the frame, and get personalized recipes with a full nutritional breakdown — powered by Gemini Vision, Groq, and a LangGraph multi-agent pipeline.

---
# SAMPLE--
<video src="https://github.com/diljain4444/food_lens_ai/blob/main/demo.mp4" controls width="700"></video>

## ✨ Features

- **Photo → Ingredients** — Upload or snap a photo of your food/fridge and Gemini 2.5 Flash detects every visible ingredient.
- **Editable Ingredient List** — Add, remove, or confirm detected ingredients before generating recipes.
- **Multi-Recipe Generation** — A Groq-powered chef agent designs 3–4 realistic dishes from your selected ingredients, complete with step-by-step instructions and quantities.
- **Automatic Nutrition Breakdown** — Every recipe is broken down ingredient-by-ingredient and matched against a nutrition database (`final.csv`) to calculate calories, protein, fats, carbohydrates, and vitamins — both per ingredient and per dish.
- **AI Chef Chat** — Ask follow-up questions about any generated recipe (substitutions, scaling, technique, dietary tweaks) and get contextual answers.
- **Polished Single-Page UI** — A dark, animated, mobile-friendly interface with an in-browser camera, ingredient chips, and a detailed nutrition modal.

---

## 🧠 How It Works (Agent Pipeline)

The backend is orchestrated as a **LangGraph** state machine with three sequential agents:

```
   Image
     │
     ▼
[Gemini 2.5 Flash] ── extract_information() ── detects visible ingredients
     │
     ▼
 User confirms / edits ingredient list (frontend)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                     LangGraph Workflow                    │
│                                                             │
│  recipe_agent  ──▶  ingredients_find_agent  ──▶  nutritional_breakdown_agent │
│  (Groq: gpt-oss-120b)   (Groq: gpt-oss-120b)        (final.csv lookup)        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
 Final recipes + nutrition data returned to frontend
```

1. **`recipe_agent`** — Given the confirmed ingredients, an LLM (Groq `openai/gpt-oss-120b`) acts as an expert chef and proposes 3–4 dishes with detailed, quantified, step-by-step instructions. Output is parsed into structured Pydantic objects.
2. **`ingredients_find_agent`** — For each generated dish, a second LLM pass extracts the *key ingredients and quantities* actually used, standardizing names (e.g. "cilantro" → "coriander leaves") and units (g, kg, tbsp, tsp, ml).
3. **`nutritional_breakdown_agent`** — Each ingredient is matched against `final.csv` (fuzzy/word-level matching with unit normalization), and per-ingredient + per-dish totals are calculated for calories, protein, fats, carbohydrates, and vitamins.
4. **`recipe_chat`** — A standalone Groq-powered chat function that takes the current dish context and a user query, allowing recipe edits, substitutions, and clarifications.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Vision / Ingredient Detection | Google Gemini 2.5 Flash (`google-genai`) |
| Recipe & Reasoning LLMs | Groq (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`) |
| Orchestration | LangChain + LangGraph (`StateGraph`) |
| Structured Output | Pydantic + `PydanticOutputParser` |
| Nutrition Database | Pandas + `final.csv` |
| Backend API | FastAPI (`my_api`) |
| Frontend | Vanilla HTML/CSS/JS single-page app (`index.html`) |

---

## 📁 Project Structure

```
food_lens_ai/
│-final.csv
|-my_api.py
|new_backend.py
|requirements.txt
|README.md
```

---

## 🔌 API Endpoints

The frontend expects the following endpoints from the backend (base URL configured in `index.html` as `API`):

### `POST /extract_ingredients`
Upload a food image and receive detected ingredients.

**Request:** `multipart/form-data` with field `image`
**Response:**
```json
{ "detected_ingredients": ["onion", "tomato", "paneer"] }
```

### `POST /main_endpoint`
Generate recipes + nutrition breakdown from a confirmed ingredient list.

**Request:**
```json
{ "final_ingredients": ["onion", "tomato", "paneer"] }
```
**Response:**
```json
{ "final_results": [ { "dish_name": "...", "full_recipe": [...], "overall_calories": 0, "...": "..." } ] }
```

### `POST /recipe_chat`
Ask a follow-up question about a specific generated dish.

**Request:**
```json
{ "dish_info": { "...": "dish object from final_results" }, "user_query": "Can I substitute paneer with tofu?" }
```
**Response:**
```json
{ "response": "Yes, here's how..." }
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>

```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the backend directory with:
```
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the backend API
```bash
cd my_api
uvicorn main:app --reload --port 8000
```
*(Replace `main:app` with the actual module:app path for your FastAPI instance.)*

### 5. Open the frontend
Simply open `index.html` in a browser, or serve it with a static file server:
```bash
python -m http.server 5500
```
Make sure the `API` constant in `index.html` points to your running backend (`http://localhost:8000` for local dev).

> ⚠️ Since the frontend and backend run on different origins during local development, ensure CORS is enabled on the FastAPI app (`fastapi.middleware.cors.CORSMiddleware`) to allow requests from your frontend's origin.

---

## 🗺️ Roadmap / Possible Improvements

- Persist user sessions/history (saved recipes, favorites)
- Allergy and dietary-preference filtering (vegan, gluten-free, keto)
- Expand `final.csv` nutrition database coverage and add fuzzy-matching confidence scores
- Cache repeated Gemini/Groq calls to reduce latency and API cost
- Add authentication for multi-user deployments
- Dockerize backend + frontend for one-command deployment

---

## 📄 License

This project is currently unlicensed. Add a `LICENSE` file (MIT, Apache 2.0, etc.) if you plan to open-source it.

---

## 🙋 Author

Built by Dil as part of an AI/ML portfolio project demonstrating multimodal AI, agentic LLM pipelines (LangChain/LangGraph), and full-stack deployment.
