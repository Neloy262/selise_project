from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from typing import List,Union
import os
import datetime

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize Semantic Chunker with HF embeddings
text_splitter = SemanticChunker(
    hf_embeddings,
    breakpoint_threshold_type="percentile"
)


def parse_docx(filepath):
    # 1. Parse (Switched to partition_docx)
    elements = partition_docx(
        filename=filepath,
        # 'strategy' is less critical for DOCX as it parses XML directly,
        # but you can leave it out or keep default behavior.
        infer_table_structure=True
    )

    # 2. Smart Chunking (Same as before)
    chunked_elements = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        overlap=50
    )

    # 3. Convert to LangChain Documents (Same logic)
    documents = []
    for element in chunked_elements:
        metadata = element.metadata.to_dict()

        # --- KEEPING YOUR FIX ---
        for key, value in list(metadata.items()):
            if isinstance(value, list):
                metadata[key] = ", ".join(map(str, value))

        metadata["category"] = element.category
        documents.append(Document(page_content=element.text, metadata=metadata))

    print(f"Created {len(documents)} context-aware chunks from DOCX.")
    return documents

def parse_pdf(filepath):
    # 1. Parse
    elements = partition_pdf(
        filename=filepath,
        strategy="fast",
        infer_table_structure=True
    )

    # 2. Smart Chunking
    chunked_elements = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        overlap=50
    )

    # 3. Convert to LangChain Documents with METADATA CLEANING
    documents = []
    for element in chunked_elements:
        metadata = element.metadata.to_dict()

        # --- THE FIX STARTS HERE ---
        # Milvus/LangChain fails on list types in metadata.
        # We convert any list values (like ["eng"]) to strings (like "eng").
        for key, value in list(metadata.items()):
            if isinstance(value, list):
                metadata[key] = ", ".join(map(str, value))
        # --- THE FIX ENDS HERE ---

        metadata["category"] = element.category
        documents.append(Document(page_content=element.text, metadata=metadata))

    print(f"Created {len(documents)} context-aware chunks.")
    return documents


def process_csv(
        file_path: str,
        chunk_size: int = 300,
        languages: Union[List[str], str] = "en"
) -> List[Document]:
    """
    Reads CSV/Excel, chunks it, and generates metadata consistent with
    strict Milvus schema (including 'last_modified').
    """
    documents = []

    # 1. Prepare Metadata Basics
    abs_path = os.path.abspath(file_path)
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)

    # Get extension (e.g., 'csv') and remove dot
    ext = os.path.splitext(file_path)[-1].lower()
    file_type_str = ext.replace('.', '')

    # 2. Get Last Modified Date (ISO Format to match Unstructured/standard schemas)
    try:
        timestamp = os.path.getmtime(abs_path)
        last_modified = datetime.datetime.fromtimestamp(timestamp).isoformat()
    except Exception:
        last_modified = ""

    # 3. Handle 'languages' (Flatten list to string if needed)
    if isinstance(languages, list):
        languages_val = ", ".join(map(str, languages))
    else:
        languages_val = str(languages)

    try:
        # --- CSV HANDLING ---
        if ext == '.csv':
            chunks_iterator = pd.read_csv(file_path, chunksize=chunk_size)

            for i, chunk_df in enumerate(chunks_iterator):
                content = chunk_df.to_csv(index=False).strip()

                doc = Document(
                    page_content=content,
                    metadata={
                        "category": "Table",

                    }
                )
                documents.append(doc)

        # --- EXCEL HANDLING ---
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            total_rows = len(df)

            for i in range(0, total_rows, chunk_size):
                chunk_df = df.iloc[i: i + chunk_size]
                content = chunk_df.to_csv(index=False).strip()

                doc = Document(
                    page_content=content,
                    metadata={

                        "category": "Table",

                    }
                )
                documents.append(doc)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

    return documents

def chunking(documents):
    docs_processed = text_splitter.split_documents(documents)
    return docs_processed


# docs = process_csv("/home/mahmud/Projects/selise_project/src/selise_project/pdf_folder/fault_data_updated.csv")
#
# for doc in docs:
#     print(doc)