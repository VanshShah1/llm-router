import chromadb
from chromadb.utils import embedding_functions

def initialize_chroma_collection():
    client = chromadb.Client()
    
    # Use all-mpnet-base-v2 model instead (more powerful)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
    
    collection = client.create_collection(
        name="question_classifier",
        embedding_function=embedding_func
    )
    
    # Add example questions with labels
    examples = [
        {"question": "What is 2+2?", "label": "easy"},
        {"question": "Capital of France?", "label": "easy"},
        {"question": "Who wrote Romeo and Juliet?", "label": "easy"},
        {"question": "How to add fractions?", "label": "easy"},
        {"question": "Explain quantum entanglement", "label": "hard"},
        {"question": "Proof of Pythagoras' theorem", "label": "hard"},
        {"question": "Einstein's field equations derivation", "label": "hard"},
        {"question": "Non-Euclidean geometry applications", "label": "hard"},
    ]
    
    # Prepare collection data
    documents = [ex["question"] for ex in examples]
    metadatas = [{"label": ex["label"]} for ex in examples]
    ids = [f"id{i}" for i in range(len(examples))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def classify_question(question: str, collection, top_n: int = 3) -> str:
    # Query the collection
    results = collection.query(
        query_texts=[question],
        n_results=top_n
    )
    
    # Extract labels from metadata
    labels = [item["label"] for item in results["metadatas"][0]]
    
    # Count label occurrences
    easy_count = labels.count("easy")
    hard_count = labels.count("hard")
    
    # Determine classification
    if easy_count > hard_count:
        return "easy question"
    else:
        return "hard question"

# Initialize ChromaDB collection
collection = initialize_chroma_collection()

# Get user input and classify
user_input = input("Enter your question: ")
classification = classify_question(user_input, collection)
print(f"\nClassification: {classification}")