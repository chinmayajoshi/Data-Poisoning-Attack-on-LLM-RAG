## Data Poisioning Attack on RAG Apps

### **📚 FAISS-based Text Search:** 

A simple pipeline to chunk text, generate embeddings, and perform **semantic search** using **faiss** for fast retrieval.  

#### 🚀 **features**
- load and process `.txt` files from a directory  
- use `HuggingFace` embeddings for vector representation  
- store and retrieve chunks with **faiss**  
- persistent storage with **faiss** index + **json**  

### 👾 [TODO] **Data Poisoning Attack on RAG:** 
Create a **malicious .txt** file and inject it into the vectordb to **poison the knowledge** and the RAG App. 

#### 📜 **todo**
- [x] create streamlit app for user interaction
- [x] create poisoned .txt file  
- [ ] add poisoned .txt into vector store
- [ ] include vector store selection (base/poison) in streamlit app
- [ ] test data poisoning vulnerability 

---

### 📦 **Setup**

1. **clone the repository**  
    ```bash
    git clone https://github.com/chinmayajoshi/Data-Poisoning-Attack-on-LLM-RAG
    cd Data-Poisoning-Attack-on-LLM-RAG
    ```

2. **create a virtual environment (optional)**  
    ```bash
    python -m venv venv
    source venv/bin/activate  # windows: venv\Scripts\activate
    ```

3. **install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

---

### 📊 **Usage**

1. **add text files**: Run [create_dataset.ipynb](create_dataset.ipynb) to create a dataset from scratch.
    ```bash
    export GROQ_API_KEY="<groq-api-key>"
    ```

2. **create vectordb**:  Run [create_vectorstore.ipynb](create_vectorstore.ipynb) to generate the HuggingFace Embeddings ([intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)) and store in FAISS Vector Store.

3. **run streamlit app** Run the [app.py](app.py) python script.
    ```bash
    streamlit start app.py
    ```

---

### 🛠️ **Tech stack**
- LangChain
- FAISS
- Groq API