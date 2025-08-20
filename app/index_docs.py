import os
import requests
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_and_index_docs(docs_dir='./docs', index_url='http://localhost:5001/index', max_chunk_size=1000):
    """
    Read .txt files from the specified directory and index their contents using /index endpoint.
    """
    if not os.path.exists(docs_dir):
        logger.error(f"Directory {docs_dir} does not exist")
        return
    
    files_processed = 0
    chunks_total = 0
    chunks_indexed = 0
    files_failed = 0

    for filename in os.listdir(docs_dir):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(docs_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = create_chunks(content, filename, max_chunk_size)
            logger.info(f"Processed {filename}: {len(chunks)} chunks")
            files_processed += 1
            chunks_total += len(chunks)
            
            # Send each chunk individually to /index
            for chunk in chunks:
                try:
                    response = requests.post(index_url, json=chunk)
                    if response.status_code == 201:
                        chunks_indexed += 1
                        logger.info(f"Indexed chunk {chunk['metadata']['chunk_index']} of {filename}")
                    else:
                        logger.error(f"Failed to index chunk {chunk['metadata']['chunk_index']} of {filename}. "
                                     f"Status: {response.status_code}, Response: {response.text}")
                except Exception as e:
                    logger.error(f"Error sending chunk {chunk['metadata']['chunk_index']} of {filename}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            files_failed += 1
    
    logger.info("Indexing complete:")
    logger.info(f"  Files processed: {files_processed}")
    logger.info(f"  Files failed: {files_failed}")
    logger.info(f"  Total chunks: {chunks_total}")
    logger.info(f"  Chunks successfully indexed: {chunks_indexed}")
    logger.info(f"  Success rate: {(chunks_indexed/chunks_total*100 if chunks_total else 0):.1f}%")

def create_chunks(content: str, filename: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Split content into chunks with metadata.
    """
    chunks = []
    lines = content.split('\n')
    title = lines[0].strip() if lines else "Untitled"
    
    current_chunk = title + '\n'
    chunk_index = 0

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if len(current_chunk) + len(line) + 1 <= max_chunk_size:
            current_chunk += line + '\n'
        else:
            if len(current_chunk.strip()) > len(title):
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {
                        "filename": filename,
                        "title": title,
                        "chunk_index": chunk_index,
                        "source": "file_upload"
                    }
                })
                chunk_index += 1
            current_chunk = title + '\n' + line + '\n'
    
    if len(current_chunk.strip()) > len(title):
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {
                "filename": filename,
                "title": title,
                "chunk_index": chunk_index,
                "source": "file_upload"
            }
        })
    
    return chunks

if __name__ == "__main__":
    read_and_index_docs(
        docs_dir='./docs',
        index_url='http://localhost:5001/index',
        max_chunk_size=1000
    )
