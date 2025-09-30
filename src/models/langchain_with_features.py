import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import JSONLoader

load_dotenv()

CHUNK_DIR = Path("monthly_chunks")
SUMMARY_DIR = Path("summaries")
SUMMARY_DIR.mkdir(exist_ok=True)

# ----------------------------
# 1️⃣ Define structured output schema
# ----------------------------
class CategorySummary(BaseModel):
    category: str = Field(..., description="Category name")
    monthly_spend: float
    budget: float
    overspend_percent: float
    anomalies: list[str]
    rolling_mean: float
    rolling_std: float

class MonthSummary(BaseModel):
    month: str
    total_spend: float
    categories: list[CategorySummary]
    recommendations: str

parser = PydanticOutputParser(pydantic_object=MonthSummary)

# ----------------------------
# 2️⃣ Load JSON files as LangChain Documents
# ----------------------------
def load_month_docs():
    docs = []
    for month_file in CHUNK_DIR.iterdir():
        loader = JSONLoader(
            file_path=str(month_file),
            jq_schema=".[]",   # treat each transaction as a separate document
            text_content=False
        )
        month_docs = loader.load()
        for d in month_docs:
            d.metadata["month"] = month_file.stem  # add month metadata
        docs.extend(month_docs)
    return docs

docs = load_month_docs()

# ----------------------------
# 3️⃣ Build FAISS vector store
# ----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(docs, embeddings)

# ----------------------------
# 4️⃣ Prompt + LLM
# ----------------------------
prompt = PromptTemplate(
    input_variables=["data_json"],
    template="""
You are a financial analyst.

Analyze the transactions for a given month.

{format_instructions}

Transactions:
{data_json}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# ----------------------------
# 5️⃣ Runnable chain (retriever → prompt → llm → parser)
# ----------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

workflow = (
    {"data_json": retriever | (lambda docs: json.dumps([d.page_content for d in docs]))}
    | prompt
    | llm
    | parser
)

# ----------------------------
# 6️⃣ Run & Save Summary
# ----------------------------
def analyze_and_save_month(month_str: str):
    summary = workflow.invoke(month_str)

    # Save summary JSON
    out_path = SUMMARY_DIR / f"{month_str}.json"
    with open(out_path, "w") as f:
        json.dump(summary.dict(), f, indent=4)

    return summary

# Example usage
month_to_analyze = "2023-10"
summary = analyze_and_save_month(month_to_analyze)
print(summary.model_dump_json(indent=2))
