
This project is a Retrieval-Augmented Generation (RAG) system that enables users to upload PDF documents and receive accurate, context-aware answers to questions grounded in the source material. It integrates document parsing, semantic search, and large language model reasoning into a complete end-to-end pipeline.

When a PDF is uploaded, the system extracts text page by page using a PDF parsing library. The extracted content is then divided into smaller, overlapping chunks to preserve contextual continuity while improving retrieval precision — ensuring information is structured into segments optimized for semantic search.

Each chunk is transformed into a high-dimensional vector embedding using a pre-trained sentence transformer model. Unlike keyword-based search, these embeddings capture the deeper semantic meaning of the text. All embeddings are stored in a FAISS (Facebook AI Similarity Search) index, enabling fast and efficient nearest-neighbor similarity searches across large document collections.

When a user submits a query, it is converted into an embedding using the same model. FAISS compares the query vector against stored document vectors and retrieves the most semantically relevant chunks, which are then used as contextual input for the language model.

These retrieved chunks are passed to a large language model via the OpenRouter API, with explicit instructions to generate responses strictly from the provided context. This approach significantly reduces hallucination and improves factual reliability, ensuring answers remain grounded in the original document rather than the model's external knowledge.

To further enhance transparency, the system maps retrieved chunks back to their source PDF pages and displays them as reference images, allowing users to visually verify where each answer originates. This audit trail builds trust and accountability in the system's outputs.

Overall, this project demonstrates a fully functional RAG pipeline — combining PDF processing, embedding generation, vector-based retrieval, and LLM reasoning — into a cohesive AI application capable of delivering reliable, document-grounded responses.