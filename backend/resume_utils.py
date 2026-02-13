# backend/resume_utils.py
import os
import re
import fitz
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA

# ----------------- OCR CONFIG -----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- Load File -----------------
def load_file(file):
    text = ""
    try:
        filename = file.filename.lower()
        file.seek(0)

        if filename.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                page_text = page.get_text()
                if len(page_text.strip()) < 20:  # OCR fallback
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)

                text += clean_text(page_text) + "\n"
            doc.close()

        elif filename.endswith(".txt"):
            text = file.read().decode("utf-8")

        else:
            return None, "Unsupported file type"

    except Exception as e:
        return None, f"Error reading file: {str(e)}"

    text = re.sub(r'\s+', ' ', text).strip()
    return text, None

# ----------------- Clean OCR Noise -----------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-zA-Z0-9.,:/\-\n ]+', ' ', text)
    return text.strip()

# ----------------- Split Lines -----------------
def split_into_sentences(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 5]
    return lines

# ----------------- Chunk Sentences -----------------
def chunk_sentences(sentences, chunk_size=4):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(" ".join(sentences[i:i+chunk_size]))
    return chunks

# ----------------- Embeddings -----------------
def build_embeddings(sentences):
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return model, embeddings

# ----------------- Semantic Search -----------------
def retrieve_best_sentence(question, model, sentence_embeddings, sentences):
    q_emb = model.encode([question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, sentence_embeddings)
    best_idx = sims.argmax()
    confidence = float(sims[0][best_idx])
    return sentences[best_idx], round(confidence, 3)

# ----------------- Step 1: Question Router -----------------
def route_question(question: str) -> str:
    q = question.lower().strip()

    genai_keywords = [
        "summarize", "summary", "explain", "describe", "overview",
        "rewrite", "paraphrase", "profile", "bio", "about me",
        "introduction", "in simple words"
    ]

    for kw in genai_keywords:
        if kw in q:
            return "genai"

    return "semantic"

# ----------------- Step 2: Top-K Context for GenAI -----------------
def get_top_k_context(question, model, sentence_embeddings, sentences, k=5):
    q_emb = model.encode([question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, sentence_embeddings)[0]

    top_indices = sims.argsort()[-k:][::-1]
    top_sentences = [sentences[i] for i in top_indices]

    return " ".join(top_sentences)

# ----------------- Offline GenAI -----------------
hf_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

def run_genai_chain(sentences, question):
    try:
        embed_model = SentenceTransformer("all-mpnet-base-v2")
        chunks = chunk_sentences(sentences)

        vectorstore = FAISS.from_texts(chunks, embed_model)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        return qa_chain.run(question)

    except Exception as e:
        return f"Offline GenAI QA error: {str(e)}"





#
#  # backend/resume_utils.py
# import os
# import re
# import fitz
# from PIL import Image
# import pytesseract
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # LangChain (latest 1.2.x)from langchain_community.vectorstores import FAISS
# # Replace this:
# # from langchain.chains import RetrievalQA

# # With this:
# from langchain_classic.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline

# from transformers import pipeline

# # ----------------- OCR CONFIG -----------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # ----------------- Load File -----------------
# def load_file(file):
#     text = ""
#     try:
#         filename = file.filename.lower()
#         file.seek(0)
#         if filename.endswith(".pdf"):
#             doc = fitz.open(stream=file.read(), filetype="pdf")
#             for page in doc:
#                 page_text = page.get_text()
#                 # If page has very little text → use OCR
#                 if len(page_text.strip()) < 20:
#                     pix = page.get_pixmap(dpi=200)
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     page_text = pytesseract.image_to_string(img)
#                 # Merge OCR lines into paragraphs
#                 page_text = merge_lines(page_text)
#                 text += page_text + " "
#             doc.close()
#         elif filename.endswith(".txt"):
#             text = file.read().decode("utf-8")
#         else:
#             return None, "Unsupported file type"
#     except Exception as e:
#         return None, f"Error reading file: {str(e)}"
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text, None

# # ----------------- Merge lines -----------------
# def merge_lines(text):
#     lines = text.splitlines()
#     merged = []
#     buffer = ""
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if buffer and not buffer[-1] in ".!?:":  # merge lines in paragraph
#             buffer += " " + line
#         else:
#             if buffer:
#                 merged.append(buffer)
#             buffer = line
#     if buffer:
#         merged.append(buffer)
#     return " ".join(merged)

# # ----------------- Split Sentences & Merge -----------------
# def split_into_sentences(text, merge_words=2):
#     raw_sentences = re.split(r'(?<=[.!?])\s+', text)
#     merged = []
#     buffer = ""
#     for s in raw_sentences:
#         s = s.strip()
#         if not s:
#             continue
#         if len(s.split()) < merge_words:
#             buffer += " " + s
#         else:
#             if buffer:
#                 merged.append((buffer + " " + s).strip())
#                 buffer = ""
#             else:
#                 merged.append(s)
#     if buffer:
#         merged.append(buffer.strip())
#     return merged

# # ----------------- Embeddings -----------------
# def build_embeddings(sentences):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     embeddings = model.encode(sentences, convert_to_numpy=True)
#     return model, embeddings

# # ----------------- Semantic Search -----------------
# def retrieve_best_sentence(question, model, sentence_embeddings, sentences):
#     q = re.sub(r'[^\w\s]', '', question.lower())
#     question_emb = model.encode([q], convert_to_numpy=True)
#     sims = cosine_similarity(question_emb, sentence_embeddings)
#     best_idx = sims.argmax()
#     best_sentence = sentences[best_idx]
#     confidence = (sims[0][best_idx] + 1) / 2
#     return best_sentence, confidence

# # ----------------- Offline GenAI Summarizer / QA -----------------
# hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

# def genai_summarize(text):
#     if not text.strip():
#         return ""
#     words = text.split()
#     chunks = []
#     chunk = []
#     for w in words:
#         chunk.append(w)
#         if len(chunk) >= 1000:
#             chunks.append(" ".join(chunk))
#             chunk = []
#     if chunk:
#         chunks.append(" ".join(chunk))
    
#     summaries = []
#     for c in chunks:
#         summary = hf_summarizer(c, max_length=150, min_length=60, do_sample=False)[0]["summary_text"]
#         summaries.append(summary)
#     return " ".join(summaries)

# # ----------------- GenAI QA (new unified function) -----------------
# def run_genai_chain(sentences, question):
#     """
#     Perform question-answering on local text using LangChain + FAISS + HuggingFace
#     """
#     try:
#         # Build FAISS index from sentences
#         index = FAISS.from_texts(sentences, SentenceTransformer('all-mpnet-base-v2'))
#         # Wrap HuggingFace summarization pipeline as LLM
#         llm = HuggingFacePipeline(pipeline=hf_summarizer)
#         # Create RetrievalQA chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=index.as_retriever()
#         )
#         return qa_chain.run(question)
#     except Exception as e:
#         return f"Offline GenAI QA not available: {str(e)}"








# some erros  # backend/resume_utils.py
# import os
# import re
# import fitz
# from PIL import Image
# import pytesseract
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Optional: LangChain / HuggingFace (offline)
# from langchain_classic.chains import RetrievalQA          # if you use classic chains
# from langchain_community.vectorstores import FAISS        # new recommended vectorstore
# from langchain_community.llms import HuggingFacePipeline # new location for HFPipeline

# from transformers import pipeline

# # ----------------- OCR CONFIG -----------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # ----------------- Load File -----------------
# def load_file(file):
#     text = ""
#     try:
#         filename = file.filename.lower()
#         file.seek(0)
#         if filename.endswith(".pdf"):
#             doc = fitz.open(stream=file.read(), filetype="pdf")
#             for page in doc:
#                 page_text = page.get_text()
#                 # If page has very little text → use OCR
#                 if len(page_text.strip()) < 20:
#                     pix = page.get_pixmap(dpi=200)
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     page_text = pytesseract.image_to_string(img)
#                 # Merge OCR lines into paragraphs
#                 page_text = merge_lines(page_text)
#                 text += page_text + " "
#             doc.close()
#         elif filename.endswith(".txt"):
#             text = file.read().decode("utf-8")
#         else:
#             return None, "Unsupported file type"
#     except Exception as e:
#         return None, f"Error reading file: {str(e)}"
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text, None

# # ----------------- Merge lines -----------------
# def merge_lines(text):
#     lines = text.splitlines()
#     merged = []
#     buffer = ""
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if buffer and not buffer[-1] in ".!?:":  # merge lines in paragraph
#             buffer += " " + line
#         else:
#             if buffer:
#                 merged.append(buffer)
#             buffer = line
#     if buffer:
#         merged.append(buffer)
#     return " ".join(merged)

# # ----------------- Split Sentences & Merge -----------------
# def split_into_sentences(text, merge_words=2):
#     raw_sentences = re.split(r'(?<=[.!?])\s+', text)
#     merged = []
#     buffer = ""
#     for s in raw_sentences:
#         s = s.strip()
#         if not s:
#             continue
#         if len(s.split()) < merge_words:
#             buffer += " " + s
#         else:
#             if buffer:
#                 merged.append((buffer + " " + s).strip())
#                 buffer = ""
#             else:
#                 merged.append(s)
#     if buffer:
#         merged.append(buffer.strip())
#     return merged

# # ----------------- Embeddings -----------------
# def build_embeddings(sentences):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     embeddings = model.encode(sentences, convert_to_numpy=True)
#     return model, embeddings

# # ----------------- Semantic Search -----------------
# def retrieve_best_sentence(question, model, sentence_embeddings, sentences):
#     q = re.sub(r'[^\w\s]', '', question.lower())
#     question_emb = model.encode([q], convert_to_numpy=True)
#     sims = cosine_similarity(question_emb, sentence_embeddings)
#     best_idx = sims.argmax()
#     best_sentence = sentences[best_idx]
#     confidence = (sims[0][best_idx] + 1) / 2
#     return best_sentence, confidence

# # ----------------- Offline GenAI Summarizer / QA -----------------
# # Uses HuggingFace distilBART for summarization locally
# hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

# def genai_summarize(text):
#     """
#     Summarize text using local HuggingFace model (distilBART)
#     """
#     if not text.strip():
#         return ""
#     # Split large text into chunks (max 1000 words)
#     words = text.split()
#     chunks = []
#     chunk = []
#     for w in words:
#         chunk.append(w)
#         if len(chunk) >= 1000:
#             chunks.append(" ".join(chunk))
#             chunk = []
#     if chunk:
#         chunks.append(" ".join(chunk))
    
#     summaries = []
#     for c in chunks:
#         summary = hf_summarizer(c, max_length=150, min_length=60, do_sample=False)[0]["summary_text"]
#         summaries.append(summary)
#     return " ".join(summaries)

# def run_offline_qa(sentences, question):
#     """
#     Perform question-answering on local text using LangChain + FAISS
#     """
#     try:
#         index = FAISS.from_texts(sentences, SentenceTransformer('all-mpnet-base-v2'))
#         llm = HuggingFacePipeline(pipeline=hf_summarizer)
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=index.as_retriever()
#         )
#         return qa_chain.run(question)
#     except Exception as e:
#         return f"Offline GenAI QA not available: {str(e)}"













# # backend/resume_utils.py
# import os
# import re
# import fitz
# from PIL import Image
# import pytesseract
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Optional: LangChain / OpenAI
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from langchain.vectorstores import FAISS

# # ----------------- OCR CONFIG -----------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # ----------------- Load File -----------------
# def load_file(file):
#     text = ""
#     try:
#         filename = file.filename.lower()
#         file.seek(0)
#         if filename.endswith(".pdf"):
#             doc = fitz.open(stream=file.read(), filetype="pdf")
#             for page in doc:
#                 page_text = page.get_text()
#                 # If page has very little text → use OCR
#                 if len(page_text.strip()) < 20:
#                     pix = page.get_pixmap(dpi=200)
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     page_text = pytesseract.image_to_string(img)
#                 # Merge OCR lines into paragraphs for better sentence extraction
#                 page_text = merge_lines(page_text)
#                 text += page_text + " "
#             doc.close()
#         elif filename.endswith(".txt"):
#             text = file.read().decode("utf-8")
#         else:
#             return None, "Unsupported file type"
#     except Exception as e:
#         return None, f"Error reading file: {str(e)}"
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text, None

# # ----------------- Merge lines -----------------
# def merge_lines(text):
#     """
#     Merge lines that are part of the same paragraph or list
#     (e.g., Skills section in scanned PDFs)
#     """
#     lines = text.splitlines()
#     merged = []
#     buffer = ""
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         # If buffer is not empty and last char is not punctuation → merge
#         if buffer and not buffer[-1] in ".!?:":
#             buffer += " " + line
#         else:
#             if buffer:
#                 merged.append(buffer)
#             buffer = line
#     if buffer:
#         merged.append(buffer)
#     return " ".join(merged)

# # ----------------- Split Sentences & Merge -----------------
# def split_into_sentences(text, merge_words=2):
#     """
#     Split text into sentences and merge small ones together.
#     merge_words: minimum number of words per sentence block
#     """
#     raw_sentences = re.split(r'(?<=[.!?])\s+', text)
#     merged = []
#     buffer = ""
#     for s in raw_sentences:
#         s = s.strip()
#         if not s:
#             continue
#         # Merge short sentences
#         if len(s.split()) < merge_words:
#             buffer += " " + s
#         else:
#             if buffer:
#                 merged.append((buffer + " " + s).strip())
#                 buffer = ""
#             else:
#                 merged.append(s)
#     # Append remaining buffer
#     if buffer:
#         merged.append(buffer.strip())
#     return merged


# # ----------------- Embeddings -----------------
# def build_embeddings(sentences):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     embeddings = model.encode(sentences, convert_to_numpy=True)
#     return model, embeddings

# # ----------------- Semantic Search -----------------
# def retrieve_best_sentence(question, model, sentence_embeddings, sentences):
#     q = re.sub(r'[^\w\s]', '', question.lower())
#     question_emb = model.encode([q], convert_to_numpy=True)
#     sims = cosine_similarity(question_emb, sentence_embeddings)
#     best_idx = sims.argmax()
#     best_sentence = sentences[best_idx]
#     confidence = (sims[0][best_idx] + 1) / 2
#     return best_sentence, confidence

# # ----------------- GenAI / LangChain -----------------
# def run_genai_chain(sentences, question):
#     try:
#         index = FAISS.from_texts(sentences, SentenceTransformer('all-mpnet-base-v2'))
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=OpenAI(),
#             retriever=index.as_retriever()
#         )
#         return qa_chain.run(question)
#     except:
#         return "GenAI answer not available (API key required)"





# # this onne was without ocr  backend/resume_utils.py
# import os
# import re
# import fitz
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Optional: LangChain / OpenAI
# # backend/resume_utils.py
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from langchain.vectorstores import FAISS

# # ----------------- Load File -----------------
# def load_file(file):
#     text = ""
#     try:
#         filename = file.filename.lower()
#         file.seek(0)
#         if filename.endswith(".pdf"):
#             doc = fitz.open(stream=file.read(), filetype="pdf")
#             for page in doc:
#                 text += page.get_text()
#             doc.close()
#         elif filename.endswith(".txt"):
#             text = file.read().decode("utf-8")
#         else:
#             return None, "Unsupported file type"
#     except Exception as e:
#         return None, f"Error reading file: {str(e)}"
#     text = re.sub(r'\s+', ' ', text)
#     return text, None

# # ----------------- Split Sentences -----------------
# # ----------------- Split Sentences (Merged Lines) -----------------
# def split_into_sentences(text):
#     # Split by line breaks first
#     lines = text.split("\n")
#     merged_lines = []
#     buffer = ""
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if buffer:
#             buffer += " " + line
#         else:
#             buffer = line
#         # If line ends with punctuation or a list number, commit buffer
#         if re.search(r'[.!?:]$|^\d+\.', line):
#             merged_lines.append(buffer)
#             buffer = ""
#     if buffer:
#         merged_lines.append(buffer)
#     return [s.strip() for s in merged_lines if s.strip()]


# # ----------------- Embeddings -----------------
# def build_embeddings(sentences):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     embeddings = model.encode(sentences, convert_to_numpy=True)
#     return model, embeddings

# # ----------------- Semantic Search -----------------
# def retrieve_best_sentence(question, model, sentence_embeddings, sentences):
#     q = re.sub(r'[^\w\s]', '', question.lower())
#     question_emb = model.encode([q], convert_to_numpy=True)
#     sims = cosine_similarity(question_emb, sentence_embeddings)
#     best_idx = sims.argmax()
#     best_sentence = sentences[best_idx]
#     confidence = (sims[0][best_idx] + 1) / 2
#     return best_sentence, confidence

# # ----------------- GenAI / LangChain -----------------
# def run_genai_chain(sentences, question):
#     try:
#         # Build FAISS index
#         index = FAISS.from_texts(sentences, SentenceTransformer('all-mpnet-base-v2'))
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=OpenAI(),
#             retriever=index.as_retriever()
#         )
#         return qa_chain.run(question)
#     except:
#         return "GenAI answer not available (API key required)"
