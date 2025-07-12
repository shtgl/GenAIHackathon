import os
import re
import sys
import asyncio
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import aiofiles
import aiohttp

# AutoGen imports
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo

# PDF processing imports
import PyPDF2
import io

# Logging imports
from log_utility import setup_logger

load_dotenv()

# Set up logger
logger = setup_logger("rag_service.log")

# Direct configuration without config file
CHUNK_SIZE = 1500
K_RESULTS = 3
SCORE_THRESHOLD = 0.1  # Lowered from 0.4 to be more permissive

# PDF files from fin_data folder
FIN_DATA_FOLDER = Path(__file__).parent / "fin_data"
PDF_SOURCES = [
    "Earnings Call Transcript Q2 - FY25.pdf",
    "Earnings Call Transcript Q1 - FY25  .pdf"  # Note: extra spaces in filename
]

def get_model_client():
    """Get the model client with proper error handling."""
    gemini_key = os.getenv("GEMINI_KEY")
    if not gemini_key:
        logger.error("GEMINI_KEY environment variable is required")
        raise ValueError("GEMINI_KEY environment variable is required. Please set it in your .env file or environment.")
    
    logger.info("Creating model client with Gemini 2.0 Flash")
    return OpenAIChatCompletionClient(
        model="gemini-2.0-flash", 
        api_key=gemini_key, 
        model_info=ModelInfo(
            multiple_system_messages=True,  # Allow multiple system messages
            vision=False,  # No vision needed for text
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="gemini-2.0-flash"
        )
    )


class DocumentIndexer:
    """Document indexer for AutoGen Memory with PDF support."""

    def __init__(self, memory: Memory, chunk_size: int = CHUNK_SIZE) -> None:
        self.memory = memory
        self.chunk_size = chunk_size
        logger.info(f"DocumentIndexer initialized with chunk_size: {chunk_size}")

    async def _fetch_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            logger.info(f"Reading PDF: {pdf_path}")
            print(f"Reading PDF: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                print(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text_content += page_text + "\n"
                    
                    # Debug: print first page content
                    if page_num == 0:
                        logger.info(f"First page text length: {len(page_text)} characters")
                        print(f"First page text length: {len(page_text)} characters")
                        if len(page_text.strip()) == 0:
                            logger.warning("No text extracted from first page")
                            print("⚠️  Warning: No text extracted from first page")
                        else:
                            logger.info(f"First page text preview: {page_text[:100]}...")
                            print(f"✅ First page text preview: {page_text[:100]}...")
                
                logger.info(f"Total extracted text length: {len(text_content)} characters")
                print(f"Total extracted text length: {len(text_content)} characters")
                return text_content
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

    async def _fetch_content(self, source: str) -> str:
        """Fetch content from URL, file, or PDF."""
        if source.startswith(("http://", "https://")):
            logger.info(f"Fetching content from URL: {source}")
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    return await response.text()
        elif source.endswith('.pdf'):
            # Handle PDF files
            pdf_path = FIN_DATA_FOLDER / source
            if pdf_path.exists():
                logger.info(f"PDF file found: {pdf_path}")
                print(f"✅ PDF file found: {pdf_path}")
                return await self._fetch_pdf_content(str(pdf_path))
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
                print(f"❌ PDF file not found: {pdf_path}")
                # Try to find the file with different name variations
                possible_names = [
                    source,
                    source.strip(),
                    source.replace("  ", " "),  # Remove double spaces
                    source.replace(" ", ""),     # Remove all spaces
                ]
                
                for name in possible_names:
                    test_path = FIN_DATA_FOLDER / name
                    if test_path.exists():
                        logger.info(f"Found PDF with corrected name: {test_path}")
                        print(f"✅ Found PDF with corrected name: {test_path}")
                        return await self._fetch_pdf_content(str(test_path))
                
                logger.error(f"Could not find PDF file: {source}")
                print(f"❌ Could not find PDF file: {source}")
                return ""
        else:
            logger.info(f"Reading file: {source}")
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                return await f.read()

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size // 2):  # 50% overlap
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0
        logger.info(f"Starting to index {len(sources)} documents")

        for source in sources:
            try:
                logger.info(f"Processing {source}...")
                print(f"Processing {source}...")
                content = await self._fetch_content(source)

                if not content:
                    logger.warning(f"No content extracted from {source}")
                    print(f"❌ No content extracted from {source}")
                    continue

                if len(content.strip()) == 0:
                    logger.warning(f"Empty content from {source}")
                    print(f"❌ Empty content from {source}")
                    continue

                logger.info(f"Extracted {len(content)} characters from {source}")
                print(f"✅ Extracted {len(content)} characters from {source}")

                # Strip HTML if content appears to be HTML
                if "<" in content and ">" in content:
                    content = self._strip_html(content)

                chunks = self._split_text(content)
                logger.info(f"Created {len(chunks)} chunks from {source}")
                print(f"✅ Created {len(chunks)} chunks from {source}")

                for i, chunk in enumerate(chunks):
                    await self.memory.add(
                        MemoryContent(
                            content=chunk, 
                            mime_type=MemoryMimeType.TEXT, 
                            metadata={"source": source, "chunk_index": i}
                        )
                    )

                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Error indexing {source}: {str(e)}")
                print(f"❌ Error indexing {source}: {str(e)}")

        logger.info(f"Total chunks indexed: {total_chunks}")
        return total_chunks


class SimpleRAGService:
    """Simplified RAG service that avoids system message conflicts."""
    
    def __init__(self):
        self.rag_memory: Optional[ChromaDBVectorMemory] = None
        self.rag_assistant: Optional[AssistantAgent] = None
        self._initialized = False
        self._model_client = None
        logger.info("SimpleRAGService initialized")
    
    def _get_model_client(self):
        """Get or create the model client."""
        if self._model_client is None:
            self._model_client = get_model_client()
        return self._model_client
    
    async def initialize(self):
        """Initialize the RAG service with AutoGen ChromaDB memory."""
        if self._initialized:
            return
        
        logger.info("Initializing AutoGen RAG service with PDF documents...")
        
        try:
            # Create persistence directory
            persistence_path = os.path.join(str(Path.home()), ".chromadb_autogen")
            os.makedirs(persistence_path, exist_ok=True)
            
            # Initialize AutoGen ChromaDB memory
            self.rag_memory = ChromaDBVectorMemory(
                config=PersistentChromaDBVectorMemoryConfig(
                    collection_name="financial_docs",
                    persistence_path=persistence_path,
                    k=K_RESULTS,
                    score_threshold=SCORE_THRESHOLD,
                )
            )
            
            logger.info(f"ChromaDB memory initialized with collection: financial_docs")
            
            # Clear existing memory
            await self.rag_memory.clear()
            logger.info("Cleared existing memory")
            
            # Index documents
            await self._index_documents()
            
            # Create RAG assistant agent WITH memory to access indexed content
            self.rag_assistant = AssistantAgent(
                name="rag_assistant", 
                model_client=self._get_model_client(),
                memory=[self.rag_memory],  # Add memory back to the assistant
                system_message="You are a helpful assistant that answers questions about financial documents and earnings call transcripts. Use the retrieved memory content to provide accurate and helpful responses."
            )
            
            self._initialized = True
            logger.info("AutoGen RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AutoGen RAG service: {e}")
            raise
    
    async def _index_documents(self):
        """Index PDF documents into the vector database."""
        if self.rag_memory is None:
            raise RuntimeError("RAG memory not initialized")
        
        logger.info("Indexing PDF documents...")
        indexer = DocumentIndexer(memory=self.rag_memory)
        chunks = await indexer.index_documents(PDF_SOURCES)
        logger.info(f"Indexed {chunks} chunks from {len(PDF_SOURCES)} PDF documents")
    
    async def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self._initialized:
            await self.initialize()
        
        if self.rag_assistant is None:
            raise RuntimeError("RAG assistant not initialized")
        
        try:
            logger.info(f"🔍 Querying: {question}")
            
            # Test memory retrieval directly
            if self.rag_memory:
                logger.info("📚 Testing memory retrieval...")
                print("📚 Testing memory retrieval...")
                memory_content = await self.rag_memory.query(question)
                logger.info(f"📚 Memory returned results")
                print(f"📚 Memory returned results")
                
                if memory_content:
                    logger.info("📝 Memory content preview:")
                    print("📝 Memory content preview:")
                    # Handle MemoryQueryResult structure
                    if hasattr(memory_content, 'content') and memory_content.content:
                        for i, content in enumerate(memory_content.content[:2]):
                            logger.info(f"  {i+1}: {content.content[:100]}...")
                            print(f"  {i+1}: {content.content[:100]}...")
                    else:
                        logger.warning(f"  Memory content structure: {type(memory_content)}")
                        print(f"  Memory content structure: {type(memory_content)}")
                else:
                    logger.warning("⚠️  No memory content found")
                    print("⚠️  No memory content found")
            
            # Get response from RAG assistant
            logger.info("🤖 Getting response from assistant...")
            print("🤖 Getting response from assistant...")
            result = await self.rag_assistant.run(task=question)
            
            # Extract only the main content from the result
            if hasattr(result, 'messages') and result.messages:
                # Get the last message which should be the assistant's response
                for message in reversed(result.messages):
                    # Check if it's a text message with content
                    if hasattr(message, 'type') and message.type == 'TextMessage':
                        if hasattr(message, 'content') and message.content:
                            response = str(message.content)
                            logger.info(f"✅ Response length: {len(response)} characters")
                            print(f"✅ Response length: {len(response)} characters")
                            return response
            
            # Fallback: try to get any content from the result
            if hasattr(result, 'content'):
                response = str(result.content)
                logger.info(f"✅ Fallback response length: {len(response)} characters")
                print(f"✅ Fallback response length: {len(response)} characters")
                return response
            
            # Final fallback to string representation
            response = str(result)
            logger.info(f"✅ Final fallback response length: {len(response)} characters")
            print(f"✅ Final fallback response length: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in RAG query: {e}")
            print(f"❌ Error in RAG query: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    async def close(self):
        """Close the RAG service and clean up resources."""
        logger.info("Closing RAG service and cleaning up resources")
        if self.rag_memory:
            await self.rag_memory.close()
        self._initialized = False
        logger.info("RAG service closed successfully")


# Global RAG service instance
rag_service = SimpleRAGService()
logger.info("Global RAG service instance created") 