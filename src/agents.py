"""LangGraph multi-agent workflow for the knowledge platform."""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    RETRIEVAL_USER_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    RESPONSE_USER_PROMPT
)


class KnowledgeState(TypedDict):
    """State schema for the knowledge graph."""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    conversation_history: List[Dict[str, str]]
    documents_processed: List[str]
    error: Optional[str]


class KnowledgeNexusAgent:
    """Multi-agent system for enterprise knowledge management."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        document_processor: DocumentProcessor,
        model_name: str = "llama-3.3-70b-versatile"
    ):
        self.vector_store = vector_store
        self.document_processor = document_processor

        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            openai_api_key=api_key,
            base_url=base_url
        )
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(KnowledgeState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retrieve")

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _retrieve_node(self, state: KnowledgeState) -> KnowledgeState:
        """Retrieve relevant documents from vector store."""
        query = state["query"]

        try:
            docs = self.vector_store.similarity_search_with_mmr(
                query=query,
                n_results=5,
                fetch_k=10,
                lambda_mult=0.5
            )
            state["retrieved_docs"] = docs
            state["error"] = None
        except Exception as e:
            state["retrieved_docs"] = []
            state["error"] = f"Retrieval error: {str(e)}"

        return state

    def _generate_node(self, state: KnowledgeState) -> KnowledgeState:
        """Generate response with citations."""
        query = state["query"]
        docs = state["retrieved_docs"]

        if not docs:
            state["response"] = "No relevant documents found. Please upload documents first or try a different query."
            return state

        context = self._format_context(docs)

        messages = [
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=RESPONSE_USER_PROMPT.format(
                question=query,
                context=context
            ))
        ]

        try:
            response = self.llm.invoke(messages)
            state["response"] = response.content
            state["error"] = None
        except Exception as e:
            state["response"] = f"Error generating response: {str(e)}"
            state["error"] = str(e)

        return state

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for prompt."""
        formatted = []

        for i, doc in enumerate(docs, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            text = doc.get('text', '')
            # Increase limit to include more content (1500 chars ~ 300-400 words)
            formatted.append(f"[{i}] Source: {source}\n{text[:1500]}")

        return "\n\n".join(formatted)

    def ingest_document(self, file_path: str, original_filename: str = None) -> Dict[str, Any]:
        """Ingest a document into the vector store."""
        try:
            chunks = self.document_processor.process_file(file_path, original_filename)
            self.vector_store.add_documents(chunks)

            return {
                "status": "success",
                "chunks": len(chunks),
                "file": original_filename or file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "file": original_filename or file_path
            }

    def query(self, query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a query through the multi-agent workflow."""
        initial_state: KnowledgeState = {
            "query": query,
            "retrieved_docs": [],
            "response": "",
            "conversation_history": history or [],
            "documents_processed": [],
            "error": None
        }

        result = self._graph.invoke(initial_state)

        return {
            "query": result["query"],
            "response": result["response"],
            "retrieved_docs": result["retrieved_docs"],
            "error": result.get("error")
        }

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            info = self.vector_store.get_collection_info()
            return {
                "status": "ready",
                "documents_indexed": info['count'],
                "collection": info['name']
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base."""
        try:
            self.vector_store.clear_collection()
            return {"status": "success", "message": "Knowledge base cleared"}
        except Exception as e:
            return {"status": "error", "error": str(e)}