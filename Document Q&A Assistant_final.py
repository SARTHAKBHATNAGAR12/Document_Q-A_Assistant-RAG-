import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print("⚙️ Setting up the system...")

# 2. Load Local Embeddings and Vector Database
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./sapiens_hybrid_db", 
    embedding_function=hf_embeddings
)

# 3. Setup the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)

# 4. Define the AI Prompt (System Instructions)
system_prompt = (
    "You are a highly intelligent and helpful AI assistant named 'Sapiens Scholar'. "
    "Use the given context from the book 'Sapiens' to answer the user's question accurately. "
    "If the answer is not in the context, clearly state that you don't have enough information from the book. "
    "Do not hallucinate or use outside knowledge.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Create the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# Retrieve the top 3 most relevant chunks from the database
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n" + "="*50)
print("🎉 SAPIENS SCHOLAR AI IS READY! 🎉")
print("Ask your question (Type 'exit' to quit)")
print("="*50 + "\n")

# 6. Chat Loop (For continuous Q&A)
while True:
    user_query = input("👤 Your Question: ")
    
    # Exit condition
    if user_query.lower() in ['exit', 'quit']:
        print(" AI: It was great talking to you. Goodbye!")
        break
        
    # Ignore empty inputs
    if not user_query.strip():
        continue
        
    print(" AI is thinking...")
    
    # Trigger the AI pipeline
    response = rag_chain.invoke({"input": user_query})
    
    print("\n Answer:")
    print(response["answer"])
    print("-" * 50)