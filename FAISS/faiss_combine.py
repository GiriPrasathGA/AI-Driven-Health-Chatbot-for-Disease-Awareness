from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-light-v3.0")

# Load the first FAISS index
combined = FAISS.load_local("faiss_part1", embeddings, allow_dangerous_deserialization=True)

# Merge others
for dir_name in ["faiss_part2", "faiss_part3","faiss_part4"]:
    vs = FAISS.load_local(dir_name, embeddings, allow_dangerous_deserialization=True)
    combined.merge_from(vs)

# Save combined index
combined.save_local("faiss_medicine_index_combined")
print("✅ Combined FAISS index saved as 'faiss_medicine_index_combined'")
