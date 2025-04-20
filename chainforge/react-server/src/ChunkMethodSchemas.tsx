import { ModelSettingsDict } from "./backend/typing";

/**
 * Overlapping + LangChain
 */
export const OverlappingLangChainSchema: ModelSettingsDict = {
  fullName: "Overlapping + LangChain",
  description: "Chunk text via LangChain's RecursiveCharacterTextSplitter.",
  schema: {
    type: "object",
    required: ["chunk_size", "chunk_overlap"],
    properties: {
      chunk_size: { type: "number", default: 300, title: "Chunk Size (chars)" },
      chunk_overlap: { type: "number", default: 50, title: "Overlap (chars)" },
    },
  },
  uiSchema: {
    chunk_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 100,
        max: 2000,
        step: 50,
      },
    },
    chunk_overlap: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 500,
        step: 10,
      },
    },
  },
  postprocessors: {},
};

/**
 * Overlapping + OpenAI tiktoken
 */
export const OverlappingOpenAITiktokenSchema: ModelSettingsDict = {
  fullName: "Overlapping + OpenAI tiktoken",
  description: "Chunk text using the OpenAI tiktoken library with overlap.",
  schema: {
    type: "object",
    required: ["chunk_size", "chunk_overlap", "model_name"],
    properties: {
      chunk_size: {
        type: "number",
        default: 200,
        title: "Chunk Size (tokens)",
      },
      chunk_overlap: {
        type: "number",
        default: 50,
        title: "Overlap (tokens)",
      },
      model_name: {
        type: "string",
        default: "gpt-3.5-turbo",
        title: "OpenAI Model (for Tokenizer)",
        description: "Specifies which model's tokenizer to use (e.g., gpt-4, gpt-3.5-turbo). Affects how text is split into tokens.",
        enum: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
      }
    },
  },
  uiSchema: {
    chunk_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 50,
        max: 8192,
        step: 50,
      },
    },
    chunk_overlap: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 500,
        step: 10,
      },
    },
    model_name: {
        "ui:widget": "select",
    }
  },
  postprocessors: {},
};

/**
 * Overlapping + HuggingFace Tokenizers
 */
export const OverlappingHuggingfaceTokenizerSchema: ModelSettingsDict = {
  fullName: "Overlapping + HuggingFace Tokenizers",
  description: "Chunk text using HuggingFace tokenizer-based segmentation with overlap.",
  schema: {
    type: "object",
    required: ["model_name", "chunk_size", "chunk_overlap"],
    properties: {
      model_name: {
        type: "string",
        default: "bert-base-uncased",
        title: "HuggingFace Model (for Tokenizer)",
        description: "Identifier of the HF model whose tokenizer to use (e.g., bert-base-uncased, distilbert-base-uncased).",
        enum: ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "gpt2", "sentence-transformers/all-MiniLM-L6-v2"],
      },
      chunk_size: {
        type: "number",
        default: 200,
        title: "Chunk Size (tokens)",
      },
      chunk_overlap: {
        type: "number",
        default: 50,
        title: "Overlap (tokens)",
      },
    },
  },
  uiSchema: {
    model_name: {
      "ui:widget": "select",
       "ui:options": {
           "placeholder": "Select or type a HuggingFace model ID"
       }
    },
    chunk_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 50,
        max: 2000,
        step: 50,
      },
    },
    chunk_overlap: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 500,
        step: 10,
      },
    },
  },
  postprocessors: {},
};

/**
 * Syntax-based spaCy
 */
export const SyntaxSpacySchema: ModelSettingsDict = {
  fullName: "Syntax-based spaCy",
  description: "Splits text into sentences using a specified spaCy language model.",
  schema: {
      type: "object",
      required: ["model_name"],
      properties: {
        model_name: {
            type: "string",
            default: "en_core_web_sm",
            title: "SpaCy Language Model",
            description: "The name of the SpaCy model to use (must be installed). E.g., en_core_web_sm, en_core_web_lg, de_core_news_sm.",
            enum: [
                "en_core_web_sm",
                "en_core_web_md",
                "en_core_web_lg",
                "en_core_web_trf",
                "de_core_news_sm",
                "es_core_news_sm",
                "fr_core_news_sm",
            ],
        }
      }
  },
  uiSchema: {
      model_name: {
          "ui:widget": "select",
          "ui:options": {
              "placeholder": "Select or type an installed SpaCy model name"
          }
      }
  },
  postprocessors: {},
};

/**
 * Syntax-based TextTiling
 */
export const SyntaxTextTilingSchema: ModelSettingsDict = {
  fullName: "Syntax-based TextTiling",
  description: "Splits text into multi-sentence topical segments using NLTK's TextTiling.",
  schema: {
    type: "object",
    required: ["w", "k"],
    properties: {
      w: { type: "number", default: 20, title: "Pseudo-Sentence Block Size (w)" },
      k: { type: "number", default: 10, title: "Gap Identification Block Count (k)" },
    },
  },
  uiSchema: {
    w: {
      "ui:widget": "range",
      "ui:options": {
        min: 5,
        max: 50,
        step: 1,
      },
    },
    k: {
      "ui:widget": "range",
      "ui:options": {
        min: 2,
        max: 50,
        step: 1,
      },
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: TextTiling + spaCy
 */
export const HybridTextTilingSpacySchema: ModelSettingsDict = {
  fullName: "Hybrid: TextTiling + spaCy",
  description:
    "Combines TextTiling for broad segmentation, then spaCy for finer splits.",
  schema: {
    type: "object",
    required: ["w", "k"],
    properties: {
      w: { type: "number", default: 20, title: "Window size (w)" },
      k: { type: "number", default: 10, title: "Block comparison size (k)" },
    },
  },
  uiSchema: {
    w: {
      "ui:widget": "range",
      "ui:options": {
        min: 5,
        max: 50,
        step: 5,
      },
    },
    k: {
      "ui:widget": "range",
      "ui:options": {
        min: 5,
        max: 50,
        step: 5,
      },
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: BERTopic + spaCy
 */
export const HybridBERTopicSchema: ModelSettingsDict = {
  fullName: "BERTopic + spaCy",
  description: "Splits text using a hybrid approach with BERTopic + spaCy.",
  schema: {
    type: "object",
    required: ["min_topic_size"],
    properties: {
      min_topic_size: {
        type: "number",
        default: 2,
        title: "Min Topic Size",
      },
    },
  },
  uiSchema: {
    min_topic_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 2,
        max: 20,
        step: 1,
      },
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + Gensim
 */
export const HybridRecursiveGensimSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + Gensim",
  description: "Combines a recursive approach with Gensim's text modeling.",
  schema: {
    type: "object",
    required: ["max_words"],
    properties: {
      max_words: {
        type: "number",
        default: 300,
        title: "Words per chunk",
      },
    },
  },
  uiSchema: {
    max_words: {
      "ui:widget": "range",
      "ui:options": {
        min: 50,
        max: 2000,
        step: 50,
      },
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + Cohere
 */
export const HybridRecursiveCohereSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + Cohere",
  description: "Uses Cohere's embeddings to guide recursive chunking.",
  schema: {
    type: "object",
    required: ["max_tokens"],
    properties: {
      max_tokens: {
        type: "number",
        default: 512,
        title: "Max tokens per chunk",
      },
      threshold: {
        type: "number",
        default: 0.75,
        title: "Embedding similarity threshold",
      },
    },
  },
  uiSchema: {
    max_tokens: {
      "ui:widget": "range",
      "ui:options": {
        min: 128,
        max: 2048,
        step: 128,
      },
    },
    threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0.0,
        max: 1.0,
        step: 0.05,
      },
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + BERTopic
 */
export const HybridRecursiveBERTopicSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + BERTopic",
  description:
    "Uses a recursive approach combined with BERTopic for semantic grouping.",
  schema: {
    type: "object",
    required: ["min_topic_size", "chunk_size"],
    properties: {
      min_topic_size: {
        type: "number",
        default: 2,
        title: "Min Topic Size",
      },
      chunk_size: {
        type: "number",
        default: 300,
        title: "Base chunk size",
      },
    },
  },
  uiSchema: {
    min_topic_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 2,
        max: 20,
        step: 1,
      },
    },
    chunk_size: {
      "ui:widget": "range",
      "ui:options": {
        min: 50,
        max: 2000,
        step: 50,
      },
    },
  },
  postprocessors: {},
};

export const ChunkMethodSchemas: { [baseMethod: string]: ModelSettingsDict } = {
  overlapping_langchain: OverlappingLangChainSchema,
  overlapping_openai_tiktoken: OverlappingOpenAITiktokenSchema,
  overlapping_huggingface_tokenizers: OverlappingHuggingfaceTokenizerSchema,
  syntax_spacy: SyntaxSpacySchema,
  syntax_texttiling: SyntaxTextTilingSchema,
  hybrid_texttiling_spacy: HybridTextTilingSpacySchema,
  hybrid_bertopic_spacy: HybridBERTopicSchema,
  hybrid_recursive_gensim: HybridRecursiveGensimSchema,
  hybrid_recursive_cohere: HybridRecursiveCohereSchema,
  hybrid_recursive_bertopic: HybridRecursiveBERTopicSchema,
};

export const ChunkMethodGroups = [
  {
    label: "Overlapping Chunking",
    items: [
      {
        baseMethod: "overlapping_langchain",
        methodName: "Overlapping Chunking",
        library: "LangChain (Chars)",
        emoji: "🌐",
      },
      {
        baseMethod: "overlapping_openai_tiktoken",
        methodName: "Overlapping Chunking",
        library: "OpenAI tiktoken (Tokens)",
        emoji: "🤖",
      },
      {
        baseMethod: "overlapping_huggingface_tokenizers",
        methodName: "Overlapping Chunking",
        library: "HuggingFace (Tokens)",
        emoji: "🤗",
      },
    ],
  },
  {
    label: "Syntax-Based Chunking",
    items: [
      {
        baseMethod: "syntax_spacy",
        methodName: "Syntax-Based Chunking",
        library: "spaCy (Sentences)",
        emoji: "🐍",
      },
      {
        baseMethod: "syntax_texttiling",
        methodName: "Syntax-Based Chunking",
        library: "NLTK TextTiling (Topics)",
        emoji: "📑",
      },
    ],
  },
];