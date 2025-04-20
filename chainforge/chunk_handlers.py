import sys
from typing import List, Dict, Any, Callable
import nltk
from nltk.tokenize import TextTilingTokenizer

# SpaCy for sentence splitting and NLP objects
import spacy

# OpenAI's Tiktoken for token-based chunking
import tiktoken

# LangChain's Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFace Tokenizers for token-based chunking
from transformers import AutoTokenizer

# === Global Cache for Expensive Resources ===
# Simple cache for loaded SpaCy models to avoid reloading on every call
_spacy_models_cache: Dict[str, Any] = {}
# Cache for loaded HF tokenizers (less critical than SpaCy models, but can help)
_hf_tokenizers_cache: Dict[str, Any] = {}


# === Define the Chunking Registry (Place after imports and cache) ===
class ChunkingMethodRegistry:
    """Registry for text chunking methods."""
    _methods: Dict[str, Callable] = {}

    @classmethod
    def register(cls, identifier: str):
        """Decorator to register a chunking function."""
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Method identifier must be a non-empty string.")

        def decorator(handler_func: Callable):
            if not callable(handler_func):
                raise TypeError("Registered handler must be a callable function.")
            if identifier in cls._methods:
                 print(f"Warning: Overwriting existing chunking method '{identifier}'.", file=sys.stderr)
            cls._methods[identifier] = handler_func
            # print(f"Registered chunking method: {identifier}") # Optional: for debugging
            return handler_func
        return decorator

    @classmethod
    def get_handler(cls, identifier: str) -> Callable | None:
        """Get the handler function for a given method identifier."""
        return cls._methods.get(identifier)

# === Chunking Helper Functions ===

# --- Method 1: Overlapping Langchain ---
@ChunkingMethodRegistry.register("overlapping_langchain")
def overlapping_langchain_textsplitter(text: str, **kwargs: Any) -> List[str]:
    """
    Chunks text using LangChain's RecursiveCharacterTextSplitter.

    Kwargs:
        chunk_size (int): Target size of each chunk (default: 200).
        chunk_overlap (int): Number of characters overlapping between chunks (default: 50).
        keep_separator (bool): Whether to keep the separators in the chunks (default: True).
    """
    if not text: return [] # Handle empty input gracefully
    chunk_size = int(kwargs.get("chunk_size", 200))
    chunk_overlap = int(kwargs.get("chunk_overlap", 50))
    keep_separator = bool(kwargs.get("keep_separator", True))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator
    )
    chunks = splitter.split_text(text)
    return chunks if chunks else [text] # Return original if splitting yields nothing

# --- Method 2: Overlapping Tiktoken ---
@ChunkingMethodRegistry.register("overlapping_openai_tiktoken")
def overlapping_openai_tiktoken(text: str, **kwargs: Any) -> List[str]:
    """
    Chunks text into overlapping segments based on OpenAI's Tiktoken token count.

    Kwargs:
        chunk_size (int): Target number of tokens per chunk (default: 200).
        chunk_overlap (int): Number of tokens overlapping between chunks (default: 50).
        model_name (str): The OpenAI model name to get the tokenizer for (default: 'gpt-3.5-turbo').
                          Used to determine the correct tokenization scheme.
    """
    if not text: return [] # Handle empty input gracefully
    chunk_size = int(kwargs.get("chunk_size", 200))
    chunk_overlap = int(kwargs.get("chunk_overlap", 50))
    # --- User configurable setting ---
    model_name = kwargs.get("model_name", "gpt-3.5-turbo") # Default to common model

    try:
        # Use the user-specified model name
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError: # More specific exception for model not found
         print(f"Warning: Tiktoken encoding not found for model '{model_name}'. Falling back to 'cl100k_base'.", file=sys.stderr)
         enc = tiktoken.get_encoding("cl100k_base") # Common fallback
    except Exception as e: # Catch other potential errors
         print(f"Warning: Could not get tiktoken encoding for '{model_name}', falling back to cl100k_base. Error: {e}", file=sys.stderr)
         enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    if not tokens: return [] # Handle case where text encodes to nothing

    result = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        decoded_chunk = enc.decode(chunk_tokens).strip()
        if decoded_chunk:
             result.append(decoded_chunk)

        # Safety check: If overlap is too large, break
        if chunk_overlap >= chunk_size and chunk_size > 0:
             print(f"Warning: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Advancing by chunk_size to avoid infinite loop.", file=sys.stderr)
             start += chunk_size # Ensure progress even if overlap is invalid
             if start >= end and end < len(tokens): # If we didn't make progress, force break
                 print("Error: Failed to advance token position, breaking loop.", file=sys.stderr)
                 break
        else:
            # Normal overlap calculation
            next_start = end - chunk_overlap
            # Prevent infinite loop if no progress is made
            if next_start <= start and end < len(tokens):
                 print(f"Warning: Overlap calculation results in no progress (start={start}, next_start={next_start}). Advancing by 1 token.", file=sys.stderr)
                 start += 1 # Force minimum progress
            else:
                start = max(0, next_start) # Ensure start >= 0

        # Break if we've processed the last chunk
        if end == len(tokens):
             break

    return result if result else [text] # Return original if splitting yields nothing


# --- Method 3: Overlapping HuggingFace ---
@ChunkingMethodRegistry.register("overlapping_huggingface_tokenizers")
def overlapping_huggingface_tokenizers(text: str, **kwargs: Any) -> List[str]:
    """
    Chunks text into overlapping segments based on HuggingFace tokenizer token count.

    Kwargs:
        chunk_size (int): Target number of tokens per chunk (default: 200).
        chunk_overlap (int): Number of tokens overlapping between chunks (default: 50).
        model_name (str): The HuggingFace model name identifier to load the tokenizer
                          (default: 'bert-base-uncased').
    """
    if not text: return [] # Handle empty input gracefully
    chunk_size = int(kwargs.get("chunk_size", 200))
    chunk_overlap = int(kwargs.get("chunk_overlap", 50))
    # --- User configurable setting ---
    model_name = kwargs.get("model_name", "bert-base-uncased") # Default model

    global _hf_tokenizers_cache
    if model_name not in _hf_tokenizers_cache:
        print(f"Loading HuggingFace tokenizer: {model_name}...", file=sys.stderr)
        try:
            # Use the user-specified model name
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            _hf_tokenizers_cache[model_name] = tokenizer
            print(f"HuggingFace tokenizer '{model_name}' loaded and cached.", file=sys.stderr)
        except Exception as e: # Catch OSError, ValueError, network errors etc.
            print(f"Error loading HuggingFace tokenizer '{model_name}': {e}", file=sys.stderr)
            # Fail clearly if the specified tokenizer cannot be loaded
            raise ValueError(f"Could not load specified HuggingFace tokenizer: {model_name}") from e
    else:
        # print(f"Using cached HF tokenizer: {model_name}") # Optional debug
        tokenizer = _hf_tokenizers_cache[model_name]


    # add_special_tokens=False prevents splitting based on [CLS], [SEP] etc.
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens: return []

    result = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        # skip_special_tokens=True ensures special tokens aren't in the output text
        decoded_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if decoded_chunk:
             result.append(decoded_chunk)

        # Safety check: If overlap is too large, break
        if chunk_overlap >= chunk_size and chunk_size > 0:
            print(f"Warning: chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). Advancing by chunk_size to avoid infinite loop.", file=sys.stderr)
            start += chunk_size
            if start >= end and end < len(tokens):
                 print("Error: Failed to advance token position, breaking loop.", file=sys.stderr)
                 break
        else:
            # Normal overlap calculation
            next_start = end - chunk_overlap
            # Prevent infinite loop if no progress is made
            if next_start <= start and end < len(tokens):
                print(f"Warning: Overlap calculation results in no progress (start={start}, next_start={next_start}). Advancing by 1 token.", file=sys.stderr)
                start += 1
            else:
                start = max(0, next_start) # Ensure start >= 0


        # Break if we've processed the last chunk
        if end == len(tokens):
            break

    return result if result else [text] # Return original if splitting yields nothing


# --- Method 4: Syntax SpaCy ---
@ChunkingMethodRegistry.register("syntax_spacy")
def syntax_spacy(text: str, **kwargs: Any) -> List[str]:
    """
    Chunks text into sentences using a SpaCy language model. Caches loaded models.

    Kwargs:
        model_name (str): The SpaCy model to use for sentence splitting
                          (default: 'en_core_web_sm'). Example: 'en_core_web_lg', 'de_core_news_sm'.
    """
    if not text: return [] # Handle empty input gracefully
    # --- User configurable setting ---
    model_name = kwargs.get("model_name", "en_core_web_sm") # Default model

    global _spacy_models_cache # Use the global cache

    # Load model if not already cached
    if model_name not in _spacy_models_cache:
        print(f"Loading SpaCy model: {model_name}...", file=sys.stderr) # Info message
        try:
            _spacy_models_cache[model_name] = spacy.load(model_name)
            print(f"SpaCy model '{model_name}' loaded and cached.", file=sys.stderr)
        except OSError as e:
            # Provide helpful error message including the specified model
            print(f"Error: SpaCy model '{model_name}' not found or failed to load. \n"
                  f"Please ensure it's installed (e.g., run 'python -m spacy download {model_name}'). \n"
                  f"Original error: {e}", file=sys.stderr)
            # Fail clearly if the model cannot be loaded
            raise ValueError(f"Required SpaCy language model '{model_name}' is not available.") from e
    # else:
        # Optional: print(f"Using cached SpaCy model: {model_name}")

    # Use the (potentially cached) model
    nlp = _spacy_models_cache[model_name]

    try:
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        return sents if sents else [text] # Return original if splitting yields nothing
    except Exception as e:
        print(f"Error processing text with SpaCy model '{model_name}': {e}", file=sys.stderr)
        # Depending on desired robustness, could return [text] or raise
        raise RuntimeError(f"SpaCy processing failed for model '{model_name}'") from e


# --- Method 5: Syntax TextTiling ---
@ChunkingMethodRegistry.register("syntax_texttiling")
def syntax_texttiling(text: str, **kwargs: Any) -> List[str]:
    """
    Chunks text into topical segments using NLTK's TextTilingTokenizer.

    Kwargs:
        w (int): Size of token pseudo-sentence blocks (default: 20). Affects granularity.
        k (int): Number of blocks used for gap identification score (default: 10). Affects sensitivity.
        # smoothing_rounds (int): Number of smoothing rounds for scores (default: 1).
        # smoothing_width (int): Width of smoothing window (default: 2).
        # Can add smoothing params later if needed for finer MVP tuning.
    """
    if not text: return [] # Handle empty input gracefully
    # --- User configurable settings (MVP essentials) ---
    # Defaults match NLTK's TextTilingTokenizer internal defaults
    w = int(kwargs.get("w", 20))
    k = int(kwargs.get("k", 10))
    # smoothing_rounds = int(kwargs.get("smoothing_rounds", 1))
    # smoothing_width = int(kwargs.get("smoothing_width", 2))

    try:
        # Ensure necessary NLTK data is downloaded (punkt is often needed)
        try:
            nltk.data.find('tokenizers/punkt')
        except (LookupError, nltk.downloader.DownloadError): # Catch LookupError too
            print("NLTK 'punkt' data not found. Attempting download...", file=sys.stderr)
            try:
                nltk.download('punkt', quiet=True)
                print("NLTK 'punkt' downloaded successfully.", file=sys.stderr)
            except Exception as download_err:
                print(f"Failed to download NLTK 'punkt' data: {download_err}", file=sys.stderr)
                # Fail clearly if essential data is missing
                raise RuntimeError("NLTK 'punkt' tokenizer data is required but could not be downloaded.") from download_err

        # Initialize tokenizer with user-provided settings
        tt = TextTilingTokenizer(
            w=w,
            k=k,
            # smoothing_rounds=smoothing_rounds,
            # smoothing_width=smoothing_width
            )
        chunks = tt.tokenize(text)
        # TextTiling can sometimes return empty strings or just whitespace
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and not chunk.isspace()]
        return cleaned_chunks if cleaned_chunks else [text] # Return original if splitting yields nothing useful

    except ImportError:
         print("NLTK library not found. Please install it (`pip install nltk`).", file=sys.stderr)
         raise ValueError("NLTK TextTilingTokenizer unavailable due to missing library.")
    except Exception as e:
         # Catch-all for other unexpected NLTK errors
         print(f"An unexpected error occurred during NLTK TextTiling: {e}", file=sys.stderr)
         raise RuntimeError("TextTiling processing failed.") from e