import os
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
import matplotlib.pyplot as plt
import docx 
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("❌ Missing TOGETHER_API_KEY in .env file.")

# === Step 1: Configure your Together API key ===
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["LITELLM_API_KEY"] = TOGETHER_API_KEY
os.environ["LITELLM_PROVIDER"] = "together_ai"

# === Step 2: Set your preferred TogetherAI model ===
llm = LLM(
    model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # or "mistralai/Mixtral-8x7B-Instruct-v0.1"
    provider="together_ai",
    config={
        "api_key": TOGETHER_API_KEY,
        "temperature": 0.7,
        "max_tokens": 1024
    }
)

# === Step 3: Define the Agent ===
analyst = Agent(
    role="Senior Data Analyst",
    goal="Analyze structured datasets and provide insights",
    backstory="A world-class analyst known for uncovering patterns in messy data.",
    llm=llm,
    verbose=True
)

# === Step 4: Get user input ===
def read_file(path: str) -> tuple[str, pd.DataFrame | None]:
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            return df.head(10).to_string(index=False), df
        elif ext == ".xlsx":
            df = pd.read_excel(path)
            return df.head(10).to_string(index=False), df
        elif ext == ".txt":
            with open(path, 'r', encoding='utf-8') as f:
                return f.read(), None
        elif ext == ".docx":
            doc = docx.Document(path)
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif ext == ".pdf":
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)[:3000], None
        elif ext in [".png", ".jpg", ".jpeg"]:
            return pytesseract.image_to_string(Image.open(path)), None
        else:
            return "❌ Unsupported file format.", None
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", None

# === Step 5: Plot if numeric data ===
def plot_dataframe(df: pd.DataFrame):
    if df is None:
        return
    numeric = df.select_dtypes(include='number').columns
    if len(numeric) >= 2:
        df.plot(x=numeric[0], y=numeric[1])
        plt.title(f"{numeric[1]} vs {numeric[0]}")
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ Not enough numeric columns to generate a plot.")

# === Step 6: Prompt user ===
file_path = input("📂 Enter path to your file: ").strip()
question = input("🧠 What question do you want to ask about the file?\n> ").strip()

file_content, df = read_file(file_path)
print("\n📄 File Content Preview:\n", file_content[:1000])

# === Step 7: Define agent and task ===
analyst = Agent(
    role="Senior Data Analyst",
    goal="Answer user questions about uploaded documents and data",
    backstory="An expert in interpreting structured and unstructured files.",
    llm=llm,
    verbose=True
)

task = Task(
    description=(
        f"User uploaded a file with the following content:\n\n{file_content[:1500]}\n\n"
        f"The user asks:\n{question}\n\n"
        "Please provide a clear, concise, and structured answer based on the file content."
    ),
    expected_output="A well-structured summary and answer to the question.",
    agent=analyst
)

# === Step 8: Run Crew ===
crew = Crew(
    agents=[analyst],
    tasks=[task],
    process=Process.sequential
)

print("\n🧠 Running AI analysis...\n")
try:
    result = crew.kickoff()
    print("\n✅ Final Answer:\n")
    print(result)
except Exception as e:
    print(f"\n❌ Error running CrewAI: {e}")

# === Step 9: Plot if applicable ===
plot_dataframe(df)