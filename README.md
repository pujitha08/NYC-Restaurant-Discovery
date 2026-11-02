# NYC Restaurant Discovery

A Text Mining / Information Retrieval course project. Users describe their
dietary needs in natural language (e.g. "nut-free brunch in Brooklyn for a
group with a vegan friend") and get ranked NYC restaurant recommendations
backed by evidence from real reviews.

## Pipeline

1. **Query parser** – GPT-4o-mini turns the free-text query into a structured `ParsedQuery`.
2. **Retrieval** – compare a TF-IDF baseline against sentence-transformer embeddings.
3. **Evidence summarizer** – GPT-4o-mini summarizes the supporting review snippets for each result.

An allergen classifier, trained on the HuggingFace `14-allergens-v2` dataset,
supports allergen-aware matching.

**Data:** Google Local Reviews 2021 (McAuley Lab). Scope starts with Manhattan
and Brooklyn; edit `BOROUGHS` in [src/config.py](src/config.py) to expand.

## Team lanes

| Lane | Owns |
|------|------|
| Data and Retrieval | `src/data`, `src/retrieval` – loading/filtering reviews, building restaurant docs, TF-IDF and embedding retrieval |
| Classification and Embeddings | `src/classifier`, `eval` – allergen classifier, embedding experiments, evaluation |
| LLM Integration and App | `src/llm`, `app` – query parser, evidence summarizer, Streamlit UI |

Shared dataclasses live in [src/schemas.py](src/schemas.py); agree on changes there before editing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your OPENAI_API_KEY
```

Place downloaded datasets in `data/` (git-ignored).

Run the app from the repo root:

```bash
streamlit run app/streamlit_app.py
```

## Layout

```
src/        config, schemas, and pipeline modules (data, retrieval, classifier, llm)
app/        Streamlit app
eval/       evaluation scripts
notebooks/  exploration
data/       local datasets (git-ignored)
```
