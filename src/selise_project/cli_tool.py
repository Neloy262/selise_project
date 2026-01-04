

from VectorDB import VectorStoreManager
from parse_file import parse_pdf,chunking,process_csv,parse_docx
import os
import click
from typing import List, Optional, Dict, Any
from Retriever import RetrieverTool
from smolagents import AzureOpenAIModel, CodeAgent
from decouple import config

AZURE_ENDPOINT = config("AZURE_ENDPOINT")
API_KEY = config("API_KEY")
# ---------------------------------------------------------
# 3. Click CLI Application
# ---------------------------------------------------------
@click.group()
def cli():
    """RAG Utility: Ingest PDFs and Query Vector DB."""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--collection', default="rag_collection", help="Milvus collection name.")
def ingest(pdf_path, collection):
    """
    Upload a PDF, parse it, and insert into the Vector DB.
    """
    click.secho(f"Starting ingestion for: {pdf_path}", fg='green')
    _, ext = os.path.splitext(pdf_path)
    ext = ext.lower()

    try:
        # 1. Parse and Semantic Chunking
        if ext == '.pdf':
            raw_docs = parse_pdf(pdf_path)
        elif ext == '.xlsx' or ext == '.xls' or ext == '.csv':
            raw_docs = process_csv(pdf_path)
        elif ext == '.docx':
            raw_docs = parse_docx(pdf_path)
        else:
            raise Exception(f"Unsupported file extension: {ext}")

        # 2. Semantic Splitting (Refining chunks)
        final_docs = chunking(raw_docs)

        # 3. Insert into Milvus
        vsm = VectorStoreManager(collection_name=collection)
        vsm.add_documents(final_docs)

        click.secho("Ingestion complete!", fg='green', bold=True)

    except Exception as e:
        click.secho(f"Error during ingestion: {e}", fg='red')


@cli.command()
@click.argument('query_text')
@click.option('--collection', default="rag_collection", help="Milvus collection name.")
@click.option('--k', default=3, help="Number of results to return.")
def search(query_text, collection, k):
    """
    Search the Vector DB for similar context.
    """
    vsm = VectorStoreManager(collection_name=collection)
    results = vsm.search(query_text, k=k)

    click.secho("\n--- Search Results ---", fg='blue')
    for i, doc in enumerate(results):
        click.secho(f"Result {i + 1} (Source: {doc.metadata.get('source', 'unknown')}):", bold=True)
        click.echo(doc.page_content)
        click.echo("-" * 40)


@cli.command()
@click.argument('query_text')
@click.option('--collection', default="rag_collection")
def ask_agent(query_text, collection):
    """
    Ask an AI Agent a question based on the Vector DB knowledge.
    """
    # 1. Connect to existing DB
    vsm = VectorStoreManager(collection_name=collection)

    # 2. Setup Tool
    retriever_tool = RetrieverTool(vsm)

    # 3. Setup Azure Model
    # NOTE: It is best practice to load these from Environment Variables

    model = AzureOpenAIModel(
        model_id="gpt-4.1-mini",  # <--- MUST match the name in Azure Portal exactly
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        api_version="2024-08-01-preview"
    )

    # 4. Initialize Agent
    click.secho("Initializing Agent...", fg='yellow')
    agent = CodeAgent(
        tools=[retriever_tool],
        model=model,
        max_steps=4,
        verbosity_level=2,
    )

    click.secho(f"Agent running query: {query_text}\n", fg='cyan')

    # 5. Run
    try:
        response = agent.run(query_text)
        click.secho("\n--- Final Answer ---", fg='green', bold=True)
        click.echo(response)
    except Exception as e:
        click.secho(f"Agent Error: {e}", fg='red')

if __name__ == '__main__':
    cli()