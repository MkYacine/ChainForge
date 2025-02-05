import { ModelSettingsDict } from "./backend/typing";

export const PineconeSchema: ModelSettingsDict = {
  fullName: "Pinecone Vectorstore",
  description: "Cloud-based vector database with broad functionality",
  schema: {
    type: "object",
    required: ["api_key", "environment", "index_name"],
    properties: {
      api_key: {
        type: "string",
        title: "API Key",
      },
      environment: {
        type: "string",
        title: "Environment",
      },
      index_name: {
        type: "string",
        title: "Index Name",
      },
      namespace: {
        type: "string",
        title: "Namespace (Optional)",
      },
      metric: {
        type: "string",
        title: "Distance Metric",
        enum: ["cosine", "euclidean", "dotproduct"],
        default: "cosine",
      },
    },
  },
  uiSchema: {
    api_key: {
      "ui:widget": "password",
    },
    metric: {
      "ui:widget": "select",
    },
  },
  postprocessors: {},
};

export const FAISSSchema: ModelSettingsDict = {
  fullName: "FAISS Vectorstore",
  description: "Local disk-based vector database by Facebook AI",
  schema: {
    type: "object",
    required: ["store_path"],
    properties: {
      store_path: {
        type: "string",
        title: "Store Path",
        description: "Path to save/load the FAISS index",
      },
      dimension: {
        type: "number",
        title: "Vector Dimension",
        default: 1536,
      },
      metric: {
        type: "string",
        title: "Distance Metric",
        enum: ["l2", "inner_product", "cosine"],
        default: "cosine",
      },
    },
  },
  uiSchema: {
    metric: {
      "ui:widget": "select",
    },
    dimension: {
      "ui:widget": "range",
      "ui:options": {
        min: 64,
        max: 4096,
        step: 64,
      },
    },
  },
  postprocessors: {},
};

export const DocsSchema: ModelSettingsDict = {
  fullName: "Documents",
  description: "Non-vectorized document collection",
  schema: {
    type: "object",
    required: [],
    properties: {
      method: {
        type: "string",
        title: "Retrieval method",
        enum: ["TF-IDF", "Boolean Search", "Keyword Overlap"],
        default: "TF-IDF",
      },
    },
  },
  uiSchema: {
    method: {
      "ui:widget": "select",
    },
  },
  postprocessors: {},
};


export const VectorstoreSchemas: { [baseMethod: string]: ModelSettingsDict } = {
  pinecone: PineconeSchema,
  faiss: FAISSSchema,
  docs: DocsSchema
};
