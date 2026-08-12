from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,          # context window
    n_threads=8,         # set to your CPU core count
    temperature=0.1,     # low temp = more deterministic strategy outputs
    max_tokens=512,
    verbose=False,
    callbacks=[StreamingStdOutCallbackHandler()],
)

response = llm.invoke("You are an F1 strategy engineer. In one sentence, what is an undercut?")
print("\n\nDone.")