# Chroma compatibility issue resolution: Fixes compatibility issues between Chroma and SQLite.
# For more details, refer to the troubleshooting guide: https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import necessary modules and libraries
from tempfile import NamedTemporaryFile
import chainlit as cl
from chainlit.types import AskFileResponse
import chromadb
from chromadb.config import Settings
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE

# Global set to track namespaces
namespaces = set()

# Function to process the uploaded PDF file
def process_file(*, file: AskFileResponse) -> list:
    # Check if the uploaded file is a PDF; raise an error if not
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")
    
    # Create a temporary file to store the uploaded PDF content
    with NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        # Load the PDF using PDFPlumberLoader
        loader = PDFPlumberLoader(tempfile.name)
        documents = loader.load()

        # Split the loaded document into smaller chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,  # Maximum size of each chunk
            chunk_overlap=100  # Overlap between chunks
        )
        docs = text_splitter.split_documents(documents)

        # Add metadata to each chunk to identify the source
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        # Check if documents were successfully split
        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs

# Function to create a search engine using Chroma
def create_search_engine(*, file: AskFileResponse) -> VectorStore:
    # Process the file and save the documents in the user session
    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    
    # Initialize an OpenAI embedding model for encoding text
    encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Initialize a Chroma client and set client settings
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        allow_reset=True,  # Allow resetting the client
        anonymized_telemetry=False  # Disable telemetry
    )
    # Initialize the Chroma search engine with the client and settings
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()  # Reset to ensure a clean state

    # Create the search engine using documents and embeddings
    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=encoder,
        client_settings=client_settings 
    )

    return search_engine

# Function triggered when chat starts
@cl.on_chat_start
async def start():
    files = None
    # Prompt user to upload a PDF file until one is provided
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["application/pdf"],  # Only accept PDF files
            max_size_mb=20,  # Maximum file size limit
        ).send()
  
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # Create the search engine asynchronously
        search_engine = await cl.make_async(create_search_engine)(file=file)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()
        raise SystemError

    # Initialize the chat model for question-answering
    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k-0613',
        temperature=0,  # Set temperature to zero for deterministic outputs
        streaming=True  # Enable streaming responses
    )

    # Create a Retrieval QA chain to interact with the search engine
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Defines the QA chain type
        retriever=search_engine.as_retriever(max_tokens_limit=4097),  # Configures the retriever from Chroma

        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": EXAMPLE_PROMPT
        },
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    # Save the chain to the user session
    cl.user_session.set("chain", chain)

# Main function that handles user messages
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    # Use the chain to generate a response to the user's query
    response = await chain.acall(message.content, callbacks=[cb])
    answer = response["answer"]
    sources = response["sources"].strip()
    source_elements = []

    # Retrieve the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    # Add sources to the answer if available
    if sources:
        found_sources = []

        # Iterate through the sources and add them to the response message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create text elements for each found source
            source_elements.append(cl.Text(content=text, name=source_name))

        # Format the answer with the sources
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    # Send the answer and source elements to the user
    await cl.Message(content=answer, elements=source_elements).send()