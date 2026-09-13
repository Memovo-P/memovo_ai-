# Memovo AI Service Contract

**Document:** `docs/contracts/AI_CONTRACT.md`  
**Status:** Canonical  
**Scope:** Backend ↔ AI Service integration  
**Version:** 1.9  
**Last Updated:** 2026-09-07

---

## 1. Purpose

This document defines the canonical contract between the Memovo Backend and the Memovo AI Service.

It defines:

- service responsibilities and boundaries
- synchronous HTTP integration
- memory processing
- semantic memory search
- embedding and chunk requirements
- AI response validation expectations
- AI/Backend ownership boundaries
- failure behavior relevant to the integration

This document is the source of truth for the **Backend ↔ AI Service contract**.

The public application API is defined separately in:

`docs/contracts/API_CONTRACT.md`

---

## 2. Architectural Boundary

Memovo uses a strict separation between the Backend and AI Service.

```text
                    Memovo Backend
                         │
              ┌──────────┴──────────┐
              │                     │
          BullMQ / Workers      Public API
              │
              │ synchronous HTTP
              ▼
          AI Service
              │
       ┌──────┼──────────┐
       │      │          │
   Processing Search   RAG/Chat
```

### Backend owns

The Backend is responsible for:

- public API
- authentication and authorization
- application business rules
- MongoDB application persistence
- BullMQ
- workers and job lifecycle
- retries and backoff
- job orchestration
- link/page-content extraction
- external URL fetching
- SSRF protection
- AI request construction
- AI response validation
- vector persistence
- vector reconciliation
- vector cleanup
- notifications

### AI Service owns

The AI Service is responsible for AI-specific computation, including:

- memory content processing
- explicit Chat Save Note preparation
- chunk generation
- embeddings
- query embeddings
- vector retrieval/ranking for semantic search
- RAG answer generation where applicable

The AI Service does **not** own the application's source-of-truth database.

For semantic retrieval, the AI Service performs the MongoDB Atlas Vector Search operation using its read-only Atlas access. The Backend remains responsible for orchestration, request construction, response validation, final ownership validation, and all vector persistence/reconciliation/cleanup.

---

## 3. AI Service Execution Model

The AI Service is **synchronous HTTP-only**.

It does not own asynchronous job orchestration.

```text
Backend Worker
     │
     │ HTTP Request
     ▼
AI Service
     │
     │ HTTP Response
     ▼
Backend Worker
```

The Backend may invoke the AI Service from BullMQ workers.

BullMQ behavior is therefore a Backend concern:

```text
Queue
  ↓
Job
  ↓
Worker
  ↓
AI HTTP Request
  ↓
AI Response
  ↓
Validation
  ↓
Persistence / Reconciliation
```

The AI Service must not:

- create BullMQ jobs
- manage Backend job state
- manage retry policies
- own queue lifecycle
- directly mutate application MongoDB source-of-truth records

---

## 4. Intent Handling

### No AI Intent Detection

The AI Service does **not** determine whether a user wants to save or search a memory.

The operation is determined by the user's interaction with the application.

Examples:

```text
Normal Memory Chat / Search My Memories
        ↓
POST /ai/chat/memories
```

```text
Save as Note from Chat
        ↓
POST /ai/memories/prepare-note
```

Therefore:

- `intent detection` is not an AI Service responsibility.
- The Backend/client flow selects the intended operation.
- The AI Service executes the explicitly requested operation.

The AI Service must not infer a save/search operation from an arbitrary user message and switch endpoints or behaviors autonomously.

---

# 5. AI Endpoints

The AI Service exposes four synchronous HTTP endpoints for the current architecture. One of them (`/ai/memories/process`) is invoked from an asynchronous Backend worker, but the HTTP operation itself is synchronous: the Backend worker sends a request and waits for the AI response.

```text
POST /ai/memories/prepare-note
POST /ai/memories/process
POST /ai/memories/search
POST /ai/chat/memories
```

These endpoints are internal Backend ↔ AI Service interfaces.

They are not the same as the public Memovo application API.

---

## 5.1 Endpoint Responsibilities

| Endpoint | Purpose | Operation Type |
|---|---|---|
| `/ai/memories/prepare-note` | Prepare title and content for explicit Chat Save Note | Synchronous preparation |
| `/ai/memories/process` | Generate chunks and embeddings for a persisted memory | Synchronous HTTP processing, invoked by an asynchronous Backend worker |
| `/ai/memories/search` | Semantic retrieval of relevant memories | Synchronous retrieval |
| `/ai/chat/memories` | RAG answer generation for Memory Chat | Synchronous retrieval + generation |

### Distinction between `/ai/memories/search` and `/ai/chat/memories`

**`/ai/memories/search`** performs semantic memory retrieval only:
- Query embedding
- Vector search
- Similarity filtering and ranking
- Returns matching memories

**`/ai/chat/memories`** performs retrieval + RAG answer generation:
- Query embedding
- Vector search  
- Similarity filtering and ranking
- RAG answer generation grounded in retrieved memories
- Returns both the generated answer AND source memories

The `/ai/chat/memories` endpoint is required for normal Memovo Memory Chat where the user expects a natural language answer grounded in their saved memories, not just a list of matching memories.

---

# 6. Chat Save Note Preparation

## Endpoint

```http
POST /ai/memories/prepare-note
```

## Purpose

Prepares user-provided content for the explicit **Save Note** action from Memovo Chat.

This operation is synchronous and is separate from the existing asynchronous memory-processing pipeline.

The AI Service prepares:

- a title
- cleaned and organized note content

The AI Service must preserve the user's meaning and must not invent substantive information.

The AI Service must **not** generate tags for this operation.

The AI Service must not persist the Note or mutate application data.

---

## 6.1 Preparation Request

The Backend sends the raw user-provided content.

Conceptually:

```json
{
  "content": "The content I want to save as a note."
}
```

The `content` field is required by the public Chat Save Note contract.

No `conversationId` is required for this AI operation.

The operation is independent of the temporary Chat conversation.

---

## 6.2 Preparation Response

The AI Service returns the prepared Note fields required by the Backend.

Conceptually:

```json
{
  "title": "Prepared note title",
  "content": "Cleaned and organized note content"
}
```

The Backend must validate the response before creating the Note.

At minimum, the Backend validates:

- response structure
- `title`
- `content`
- required string constraints
- preservation of the expected Note creation contract

The AI Service must not return or generate application-owned fields such as:

- `userId`
- `tags`
- `createdAt`
- `updatedAt`
- persistence identifiers

The resulting Note uses `tags: []`.

---

## 6.3 Chat Save Note Flow

```text
User selects Save Note mode
   ↓
User writes / copies / pastes content
   ↓
Backend
   ↓
POST /ai/memories/prepare-note
   ↓
AI returns title + cleaned/organized content
   ↓
Backend validates AI response
   ↓
Backend creates normal Note
   ↓
Backend returns saved Note
   ↓
Backend starts existing async processing
   ↓
POST /ai/memories/process
   ↓
Chunks + embeddings
   ↓
Vector reconciliation
```

The Note is considered saved when the primary Note record is successfully created by the Backend.

The client does not wait for embedding/vector reconciliation.

If AI preparation fails or returns invalid output, the Backend must not create a malformed or incomplete Note.

This explicit Save Note operation does not introduce AI intent detection and does not turn ordinary Chat messages into memory mutations.

---

# 7. Memory Processing

## Endpoint

```http
POST /ai/memories/process
```

## Purpose

Processes a persisted Memovo memory into AI-derived chunks and embeddings.

The Backend invokes this operation after the relevant memory has been persisted and the asynchronous processing job reaches the AI stage.

Supported memory types in Sprint 1:

- `note`
- `link`

---

## 7.1 Processing Flow

### Note

```text
Create Note
   ↓
Backend persists Note
   ↓
BullMQ processing job
   ↓
Worker reloads current Note
   ↓
POST /ai/memories/process
   ↓
AI generates chunks + embeddings
   ↓
Backend validates response
   ↓
Backend reconciles vectors
   ↓
Processing completes
```

### Link

```text
Create Link
   ↓
Backend persists Link
   ↓
Link extraction / metadata job
   ↓
Backend fetches and extracts content
   ↓
Backend persists extracted content
   ↓
AI processing job
   ↓
Worker reloads current Link
   ↓
POST /ai/memories/process
   ↓
AI generates chunks + embeddings
   ↓
Backend validates response
   ↓
Backend reconciles vectors
   ↓
Processing completes
```

The worker must use the current persisted memory state when constructing the AI request.

---

# 8. Processing Request

The request represents the current memory content required by the AI processing pipeline.

## 8.1 Note Processing Request

The processing payload for a Note contains:

| Field | Type | Required | Nullable | Constraints |
|---|---|:---:|:---:|---|
| `type` | string | Yes | No | Must be `"note"` |
| `memoryId` | string | Yes | No | Non-empty memory identifier |
| `title` | string | Yes | No | 0–200 characters |
| `content` | string | Yes | No | 1–10,000 characters |
| `tags` | array | No | Yes | Maximum 20 tags, each maximum 50 characters |

Example:

```json
{
  "type": "note",
  "memoryId": "67890abcdef1234567890abc",
  "title": "MongoDB Performance Tips",
  "content": "Use indexes for frequently queried fields...",
  "tags": ["database", "performance"]
}
```

## 8.2 Link Processing Request

The processing payload for a Link contains:

| Field | Type | Required | Nullable | Constraints |
|---|---|:---:|:---:|---|
| `type` | string | Yes | No | Must be `"link"` |
| `memoryId` | string | Yes | No | Non-empty memory identifier |
| `url` | string | Yes | No | Maximum 2048 characters |
| `title` | string | Yes | No | 1–500 characters |
| `content` | string | No | Yes | Empty string allowed, maximum 1000 characters |
| `tags` | array | No | Yes | Maximum 20 tags, each maximum 50 characters |
| `source` | object | Yes | No | Source metadata (see 8.3) |
| `extractedContent` | string | No | Yes | Backend-extracted content (see 8.4) |

**Note:** The legacy field `whySaved` is **permanently removed** from Sprint 1 and is NOT part of any Backend → AI request. See Section 8.5 for details.

Example:

```json
{
  "type": "link",
  "memoryId": "12345abcdef6789012345abc",
  "url": "https://example.com/article",
  "title": "Understanding Vector Embeddings",
  "content": "Good reference for understanding embeddings",
  "tags": ["ai", "embeddings"],
  "source": {
    "sourceTitle": "Tech Blog",
    "sourceDescription": "Articles about AI and ML",
    "authorName": "Jane Doe",
    "publicationDate": "2026-09-01"
  },
  "extractedContent": "Vector embeddings are numerical representations..."
}
```

## 8.3 Link Source Object

The `source` object contains selected metadata extracted by the Backend. The object itself is required, but individual fields may be null.

The AI Service receives ONLY these source fields:

| Field | Type | Required | Nullable | Description |
|---|---|:---:|:---:|---|
| `sourceTitle` | string | Yes | Yes | Title of the source website/publication |
| `sourceDescription` | string | Yes | Yes | Description of the source |
| `authorName` | string | Yes | Yes | Content author name |
| `publicationDate` | string | Yes | Yes | Publication date (ISO 8601 or other string representation) |

**The following Link metadata fields are NOT sent to the AI Service:**
- `platform`
- `contentType`
- `thumbnailUrl`
- `canonicalUrl`

These fields are Backend-only metadata and are not part of the AI processing input.

## 8.4 Link extractedContent Field

The `extractedContent` field contains textual content extracted by the Backend from the Link's URL.

**Important architectural clarifications:**

- `extractedContent` is **Backend-generated**
- The Backend performs the URL fetch and content extraction
- The AI Service receives the extracted content when available
- The AI Service does **NOT** fetch URLs itself
- **Optional:** The field may be omitted if extraction failed or was not performed
- **Nullable:** The field may be `null` if explicitly set
- **Empty string allowed:** The field may be an empty string `""`
- **No AI-specific character limit:** Backend controls extraction, storage, and payload limits

**Field distinctions:**

| Field | Source | Purpose |
|---|---|---|
| `content` | User-provided | User's own context/notes about the Link |
| `extractedContent` | Backend-extracted | Page content extracted from the URL |
| `source.*` | Backend-extracted | Selected source metadata |

For AI semantic processing, a Link's context may include:
```
title + content + tags + source metadata + extractedContent
```

The `url` field provides contextual information but is **NOT an instruction for the AI Service to fetch the resource**. URL fetching, SSRF protection, and content extraction are exclusively Backend responsibilities.

## 8.5 Legacy Field Removal: whySaved

The field `whySaved` is **permanently removed** from Sprint 1 and is **NOT** part of any Backend → AI Service request or AI processing input.

### Removal Scope

The AI Service must NOT:
- Accept `whySaved` as a processing input field
- Include `whySaved` in embeddings
- Use `whySaved` in AI context generation
- Map `whySaved` to another field automatically

### Current Link Context Fields

For Link processing, the canonical user-provided context field is:

**`content`** - Optional, nullable, user-provided context/notes about the Link

**This is NOT replaced by:**
- `source.sourceDescription` - This is Backend-extracted source metadata, not user context
- `extractedContent` - This is Backend-extracted page content, not user context

### Field Purpose Clarification

| Field | Source | Purpose | User-Provided? |
|---|---|---|:---:|
| `content` | User input | User's own context/notes about the Link | ✅ Yes |
| `extractedContent` | Backend extraction | Page content extracted from URL | ❌ No |
| `source.sourceDescription` | Backend extraction | Description of the source website | ❌ No |
| ~~`whySaved`~~ | ~~Deprecated~~ | ~~Removed from Sprint 1~~ | ❌ N/A |

### Contract Enforcement

If any legacy implementation or documentation references `whySaved`:
- It is deprecated and must not be used
- It must not conflict with this canonical contract
- The Backend will not send it to the AI Service
- The AI Service must not expect or process it

**Effective Version:** This removal is effective as of AI Contract v1.5.

---

# 9. Processing Response

The AI Service returns AI-derived chunks.

The Backend validates the complete response before persistence.

A processing result contains:

- `memoryId`
- `chunks`

Each chunk contains:

- `chunkId`
- `chunkIndex`
- `content`
- `embedding`

Conceptually:

```json
{
  "memoryId": "memory-id",
  "chunks": [
    {
      "chunkId": "unique-chunk-id",
      "chunkIndex": 0,
      "content": "Chunk content",
      "embedding": [0.0123, -0.0456]
    }
  ]
}
```

The actual embedding array contains exactly **1024 numeric dimensions**.

---

# 10. Embedding Contract

Current embedding configuration:

```text
Model: Qwen3-Embedding
Dimensions: 1024
```

Every persisted embedding must:

- be an array
- contain exactly 1024 values
- contain numeric values
- correspond to the returned chunk content

The Backend validates AI-produced embeddings before persistence.

The AI Service produces embeddings.

The Backend owns persistence and reconciliation of those embeddings.

---

# 11. Chunk Contract

A processed memory is represented by zero or more AI-generated chunks.

Each chunk has:

| Field | Requirement |
|---|---|
| `chunkId` | Required unique chunk identifier |
| `chunkIndex` | Non-negative chunk position |
| `content` | Chunk text |
| `embedding` | Exactly 1024 numeric dimensions |

The Backend persists these chunks in the vector store as derived data.

The application's primary memory record remains the source of truth.

---

# 12. Vector Persistence Boundary

The AI Service does not own application vector persistence.

The ownership model is:

```text
AI Service
   │
   │ chunks + embeddings
   ▼
Backend
   │
   ├── validate
   ├── reconcile
   └── persist
        │
        ▼
MongoDB Atlas
memory_vectors
```

The vector collection is:

```text
memory_vectors
```

The Atlas Vector Search index is:

```text
vector_index
```

Vector search uses:

```text
cosine similarity
1024 dimensions
```

The Backend is responsible for keeping vector data synchronized with the current memory state.

---

# 13. Vector Reconciliation

When AI processing returns a new chunk set, the Backend reconciles the result against existing vectors for the memory.

Examples:

```text
Existing: A B C
New:      A B

→ Keep A
→ Keep B
→ Delete C
```

```text
Existing: A B
New:      A B C

→ Keep A
→ Keep B
→ Insert C
```

```text
Existing: A B
New:      A' B

→ Update A
→ Keep B
```

Repeated identical results must not create duplicate vector records.

Vector cleanup for hard-deleted memories is also a Backend responsibility.

---

# 14. Semantic Memory Search

## Endpoint

```http
POST /ai/memories/search
```

## Purpose

Generates the semantic search result for a user's query using the AI/vector-search pipeline.

The Backend supplies the authenticated user's identity and search query.

## 14.1 Search Request Schema

The AI Service receives:

```json
{
  "userId": "user-id",
  "query": "What did I save about ...?"
}
```

**Required fields:**
- `userId` - Authenticated user identifier (from Backend auth context, not client input)
- `query` - Search query string

**Fields NOT sent to AI:**
- ❌ `type` / `memoryType` - Memory type filter (e.g., `"note"`, `"link"`)
- ❌ `limit` - Result limit

The `userId` used for this operation comes from the authenticated Backend context and must not be trusted from an untrusted client-provided identity.

## 14.2 Memory Type Filtering Architecture

**Type filtering is exclusively a Backend responsibility.**

The AI Service:
- Does NOT receive a `type` or `memoryType` parameter
- Does NOT filter by Note vs Link during retrieval
- Performs unified semantic search across all memory types
- Returns all relevant memories regardless of type

The Backend:
- Receives the user's requested `type` filter from the public API (optional: `"note"` or `"link"`)
- Applies type filtering AFTER receiving the AI response
- Uses MongoDB classification queries to determine each memory's type
- Filters the AI results to match the requested type
- Returns only the requested type to Flutter, or both types if `type` is omitted

**When the public API omits `type`:**
- Backend returns both Note and Link results (no filtering)

**When the public API specifies `type: "note"`:**
- Backend filters AI results to return only Notes

**When the public API specifies `type: "link"`:**
- Backend filters AI results to return only Links

---

# 15. Semantic Search Responsibilities

The AI search pipeline is responsible for:

1. Generate the query embedding.
2. Search MongoDB Atlas Vector Search.
3. Apply the user ownership filter (`userId` pre-filter).
4. Retrieve the configured top candidate chunks.
5. Apply the configured similarity threshold.
6. Deduplicate results by memory.
7. Calculate the memory-level similarity score from the matching chunks.
8. Return the search result to the Backend.

**Note:** The AI Service does NOT receive or apply memory-type filters (e.g., `type: "note"` or `type: "link"`). Type filtering is a Backend responsibility performed after receiving the AI response. The AI Service performs unified semantic search across all memory types.

Current Sprint 1 semantic-search rules:

```text
Top chunks: 5
Similarity threshold: 0.75
Memory score: MAX matching chunk similarity
Deduplication: memoryId
```

The Backend performs the final application-level ownership validation before exposing results.

---

# 16. Search Response

## 16.1 Internal AI → Backend Response Contract

This section documents the **internal AI Service → Backend response** for `/ai/memories/search`.

This is NOT the public Flutter-facing API response. The Backend is responsible for mapping and enriching this internal response to construct the public API response governed by `docs/contracts/API_CONTRACT.md`.

### Response Schema

The AI Service returns:

```json
{
  "found": true,
  "results": [
    {
      "memoryId": "507f1f77bcf86cd799439013",
      "score": 0.94,
      "title": "MongoDB Performance Guide",
      "tags": ["mongodb", "performance"],
      "chunks": [
        {
          "chunkId": "chunk_001",
          "content": "MongoDB indexing strategies improve query performance..."
        }
      ]
    }
  ]
}
```

### Top-Level Fields

| Field | Type | Required | Nullable | Description |
|---|---|:---:|:---:|---|
| `found` | boolean | Yes | No | Whether relevant memories were found |
| `results` | array | Yes | No | Array of matching memory results (empty if not found) |

### Per-Result Fields

Each result object contains:

| Field | Type | Required | Nullable | Constraints |
|---|---|:---:|:---:|---|
| `memoryId` | string | Yes | No | Non-empty memory identifier (MongoDB ObjectId string) |
| `score` | number | Yes | No | Similarity score (finite, non-NaN, typically 0.0–1.0) |
| `title` | string | Yes | No | Memory title |
| `tags` | array | Yes | No | Array of tag strings (may be empty `[]`) |
| `chunks` | array | Yes | No | Array of matching chunks (may be empty `[]`) |

### Per-Chunk Fields

Each chunk object contains:

| Field | Type | Required | Nullable | Constraints |
|---|---|:---:|:---:|---|
| `chunkId` | string | Yes | No | Non-empty chunk identifier |
| `content` | string | Yes | No | Non-empty chunk text content |

### No-Match Response

When no relevant memories are found:

```json
{
  "found": false,
  "results": []
}
```

## 16.2 Field Exclusions

The AI Service must NOT return Backend-only or public-API-specific fields in this internal response, including:

- ❌ `id` (Backend maps `memoryId` → `id` for public API)
- ❌ `type` (Backend determines type via MongoDB classification)
- ❌ `content` (Backend fetches from MongoDB when needed)
- ❌ `url` (Backend fetches from MongoDB for Link memories)
- ❌ `matchedChunks` (Backend maps `chunks` → `matchedChunks` for public API)
- ❌ `similarityScore` (Backend maps `score` → `similarityScore` for public API)
- ❌ `createdAt`, `updatedAt`, `userId`, or other application metadata

These fields are either:
1. Fetched by the Backend from MongoDB based on the returned `memoryId`
2. Renamed/transformed by the Backend for the public API response
3. Computed by the Backend (e.g., type classification)

## 16.3 Backend Response Mapping

The Backend receives the internal AI response and:

1. **Validates** the response structure and required fields
2. **Deduplicates** results by `memoryId` (if not already done by AI)
3. **Applies type filtering** if requested by the public API (Backend-side, post-AI-response)
4. **Applies limit** to the final result set
5. **Fetches full memory data** from MongoDB (type, content, url, etc.)
6. **Enriches** the response with MongoDB data
7. **Renames fields** for public API compatibility:
   - `memoryId` → `id`
   - `chunks` → `matchedChunks`
   - `score` → `similarityScore`
8. **Verifies ownership** before exposing results to the user
9. **Constructs** the public Flutter-facing response per `API_CONTRACT.md`

## 16.4 Response Validation

The Backend must validate the AI response before using it:

**Top-level validation:**
- `found` is boolean
- `results` is array

**Per-result validation:**
- `memoryId` is non-empty string
- `score` is finite number (not NaN, not Infinity)
- `title` is string
- `tags` is array of strings
- `chunks` is array

**Per-chunk validation:**
- `chunkId` is non-empty string
- `content` is non-empty string

Invalid or malformed AI responses must be rejected with appropriate error handling.

The Backend must not trust AI output without validation.

## 16.5 Backend Type Filtering and Limit Application

The Backend is solely responsible for applying the public API's `type` filter and `limit` parameters.

### Type Filtering Process

1. **AI Service** returns unified search results containing both Notes and Links
2. **Backend** receives the AI response
3. **Backend** applies type filter if requested by public API:
   - Fetches both Note and Link collections from MongoDB
   - Classifies each `memoryId` by collection membership
   - Filters AI results to match the requested type
4. **Backend** applies the public API `limit` AFTER type filtering
5. **Backend** constructs the final Flutter response

### Type Filter Scenarios

| Public API `type` | Backend Behavior |
|---|---|
| Omitted / not specified | Returns both Notes and Links (no filtering) |
| `"note"` | Returns only Notes (filters out Links) |
| `"link"` | Returns only Links (filters out Notes) |

### Limit Application

The `limit` parameter is applied by the Backend AFTER:
- Receiving the AI response
- Deduplicating results
- Applying type filtering (if requested)

The AI Service does NOT receive or apply the `limit` parameter.

### Example Flow

**Public API Request:**
```json
POST /api/search/semantic
{
  "query": "database performance",
  "type": "note",
  "limit": 5
}
```

**Backend → AI Request:**
```json
POST /ai/memories/search
{
  "userId": "507f...",
  "query": "database performance"
}
```
*Note: `type` and `limit` are NOT sent to AI*

**AI → Backend Response:**
```json
{
  "found": true,
  "results": [
    {"memoryId": "id1", ...},  // Could be Note or Link
    {"memoryId": "id2", ...},  // Could be Note or Link
    {"memoryId": "id3", ...},  // Could be Note or Link
    ...
  ]
}
```

**Backend Processing:**
1. Validates AI response
2. Deduplicates by `memoryId`
3. Fetches Notes and Links from MongoDB to classify each memory
4. Filters to keep only Notes (per `type: "note"`)
5. Applies `limit: 5` to filtered results
6. Fetches full memory data
7. Enriches and renames fields
8. Constructs public API response

**Public API Response:**
```json
{
  "found": true,
  "results": [
    // Only Notes, maximum 5 results
  ]
}
```

---

# 17. Memory Chat (RAG)

## Endpoint

```http
POST /ai/chat/memories
```

## Purpose

Generates a natural language answer to the user's question, grounded exclusively in their saved Memovo memories.

This endpoint performs **Retrieval-Augmented Generation (RAG)** for normal Memovo Memory Chat.

Unlike `/ai/memories/search` which returns only matching memories, this endpoint returns a generated answer along with the source memories used to ground that answer.

---

## 17.1 Memory Chat Responsibilities

The AI Memory Chat operation is responsible for:

1. Query embedding generation
2. Semantic retrieval of relevant memories (via MongoDB Atlas Vector Search)
3. User ownership filtering (`userId` pre-filter)
4. Similarity filtering and ranking
5. RAG answer generation grounded in retrieved memories
6. Returning structured response with answer and sources

The operation must:

- Answer ONLY using information from the user's retrieved memories
- Not fabricate information from general knowledge
- Not claim information came from memories when it did not
- Return `found: false` when no relevant memories exist

---

## 17.2 Chat Request

The Backend sends:

```json
{
  "userId": "user-id",
  "message": "User's question about their memories",
  "conversationId": "optional-conversation-id",
  "history": [
    { "role": "user", "content": "Previous message" },
    { "role": "assistant", "content": "Previous response" }
  ]
}
```

### Required Fields

| Field | Type | Required | Description |
|---|---|:---:|---|
| `userId` | string | Yes | Authenticated user ID (from Backend auth context, not client input) |
| `message` | string | Yes | User's current message/question |

### Optional Fields

| Field | Type | Required | Description |
|---|---|:---:|---|
| `conversationId` | string | No | For conversation continuity (temporary, in-memory) |
| `history` | array | No | Recent conversation context (see 17.2.1) |

The `userId` must come from the authenticated Backend session and must not be trusted from arbitrary user input.

## 17.2.1 Chat History Contract

The `history` field is optional and provides conversation context to the AI Service.

### History Structure

Each history message is an object with:

| Field | Type | Required | Description |
|---|---|:---:|---|
| `role` | string | Yes | Message role (see allowed roles below) |
| `content` | string | Yes | Message content |

### Allowed Roles

Only the following roles are permitted in `history`:

- ✅ `"user"` - User messages
- ✅ `"assistant"` - AI assistant responses

**Forbidden roles:**
- ❌ `"system"` - Not allowed in history
- ❌ Any other role - Not allowed

### History Size Limits

The Backend must enforce the following limits when constructing the `history` array:

| Limit | Value | Description |
|---|---|---|
| **Maximum messages** | 30 | Maximum number of history messages |
| **Maximum per-message size** | 6,000 characters | Maximum `content` length for a single message |
| **Maximum total size** | 30,000 characters | Maximum combined length of all message `content` fields |

### Backend History Responsibilities

The Backend is solely responsible for:

1. **Validation** - Ensuring all messages have valid `role` and `content` fields
2. **Role validation** - Ensuring only `user` and `assistant` roles are included
3. **Per-message limit** - Ensuring no single message `content` exceeds 6,000 characters
4. **Total size limit** - Ensuring total of all message `content` lengths does not exceed 30,000 characters
5. **Message count limit** - Ensuring no more than 30 messages are included
6. **Trimming overflow** - Removing oldest messages first when limits are exceeded
7. **Message integrity** - Messages must never be partially truncated; a message is either included in full or excluded entirely

### History Trimming Behavior

When the stored conversation exceeds the limits, the Backend must:

1. Start with the full conversation history
2. Calculate total size and message count
3. If limits are exceeded:
   - Remove the **oldest** messages first
   - Keep the **newest** messages that fit within both limits
   - Never partially truncate a message
4. Send the trimmed history to the AI Service

**Example trimming scenario:**

```text
Stored conversation: 40 messages, 35,000 total characters

Trimming process:
1. Remove oldest message → 39 messages
2. Check total size → still exceeds 30,000 characters
3. Remove next oldest → 38 messages
4. Check total size → still exceeds 30,000 characters
5. Continue removing oldest until:
   - Message count ≤ 30 AND
   - Total size ≤ 30,000 characters
   
Result: Send only the newest X messages that fit
```

### AI Service History Expectations

The AI Service:

- ✅ Receives already-bounded history from the Backend
- ✅ Uses the provided history as conversation context
- ❌ Must NOT perform its own history trimming or validation
- ❌ Must NOT enforce size limits (Backend responsibility)
- ❌ Does NOT own conversation state

### Conversation State Ownership

- **Backend** owns conversation state and history management
- **AI Service** is stateless and does not persist conversations
- `conversationId` is used by the Backend for in-memory or temporary conversation tracking
- The AI Service uses `conversationId` for conversation continuity but does not manage conversation lifecycle

### Context Boundary Clarification

These limits (30 messages, 6,000 per message, 30,000 total) are **safety and context boundaries**.

The Backend is NOT required to send the maximum 30 messages on every request. The Backend should send only the conversation context that is relevant and fits within the limits.

**Valid history examples:**

```json
// No history (new conversation)
{
  "userId": "...",
  "message": "...",
  "history": []
}

// Short history (5 messages)
{
  "userId": "...",
  "message": "...",
  "history": [
    { "role": "user", "content": "First question" },
    { "role": "assistant", "content": "First answer" },
    { "role": "user", "content": "Follow-up question" },
    { "role": "assistant", "content": "Follow-up answer" },
    { "role": "user", "content": "Third question" }
  ]
}

// History omitted entirely
{
  "userId": "...",
  "message": "..."
}
```

## 17.2.2 Conversation ID Contract

The `conversationId` field is an optional identifier for a Backend-owned conversation.

### Conversation Ownership

- **Backend** owns conversation state and conversation persistence
- **AI Service** is stateless with respect to conversations
- **AI Service** must NOT store conversation or session state

### Conversation Flow

**First Chat request (new conversation):**

1. Client sends Chat request without `conversationId` (or with `conversationId` omitted)
2. Backend creates a new `conversationId`
3. Backend initializes conversation state
4. Backend sends request to AI Service (may include `conversationId` as identifier)
5. Backend returns `conversationId` to client according to public API contract

**Follow-up requests (existing conversation):**

1. Client sends Chat request with previously issued `conversationId`
2. Backend validates `conversationId` and user ownership
3. Backend loads conversation history
4. Backend applies finalized history limits (Section 17.2.1)
5. Backend sends bounded history to AI Service
6. Backend may include `conversationId` in AI request as identifier

### AI Service conversationId Usage

The AI Service:

- ✅ May receive `conversationId` as an identifier
- ✅ May use it for logging, debugging, or request correlation
- ❌ Must NOT use it to retrieve conversation state
- ❌ Must NOT use it to persist conversation state
- ❌ Must NOT store session data keyed by `conversationId`
- ❌ Does NOT own conversation lifecycle

**The AI Service is stateless.** All conversation state retrieval, history management, and persistence are Backend responsibilities.

### Conversation Ownership Enforcement

- A `conversationId` is bound to a specific user
- Backend must enforce conversation ownership
- A `conversationId` must NEVER allow access to another user's conversation
- Backend must validate that the authenticated user owns the requested `conversationId`

### Conversation Scope

`conversationId` is used for:

- ✅ Conversation continuity across multiple Chat requests
- ✅ History retrieval and management by Backend
- ✅ Grouping messages within a conversation

`conversationId` does NOT:

- ❌ Alter RAG retrieval behavior
- ❌ Alter semantic search behavior
- ❌ Change which memories are retrieved
- ❌ Affect memory processing or embeddings

### Architecture Summary

```text
Client
  ↓
  conversationId (if follow-up)
  ↓
Backend
  ├── Validates ownership
  ├── Loads conversation history
  ├── Applies history limits (30 messages, 6K/30K)
  ├── Constructs bounded history
  └── Sends to AI Service
       ↓
       AI Service (stateless)
       ├── Receives conversationId (optional identifier)
       ├── Receives bounded history
       ├── Does NOT retrieve conversation state
       └── Does NOT persist conversation state
```

The Backend provides all conversation context through the `history` array on each request. The AI Service does not maintain conversation state between requests.

---

## 17.3 Chat Response

The AI Service returns:

```json
{
  "found": true,
  "answer": "Natural language answer grounded in retrieved memories",
  "sources": [
    {
      "memoryId": "memory-id",
      "score": 0.89,
      "title": "Memory title",
      "chunks": [
        {
          "chunkId": "chunk-id",
          "content": "Relevant chunk content"
        }
      ]
    }
  ]
}
```

**Response structure:**

| Field | Type | Required | Description |
|---|---|:---:|---|
| `found` | boolean | Yes | Whether relevant memories were found |
| `answer` | string | Yes | Generated answer (may indicate no memories found) |
| `sources` | array | Yes | Source memories used to ground the answer |

**Per-source structure:**

| Field | Type | Required | Description |
|---|---|:---:|---|
| `memoryId` | string | Yes | Memory identifier |
| `score` | number | Yes | Similarity score (finite, non-NaN) |
| `title` | string | Yes | Memory title |
| `chunks` | array | Optional | Relevant chunks from this memory |

**Per-chunk structure (if present):**

| Field | Type | Required | Description |
|---|---|:---:|---|
| `chunkId` | string | Yes | Chunk identifier |
| `content` | string | Yes | Chunk text content |

---

## 17.4 Backend Validation

The Backend must validate the AI response before using it:

1. **Top-level**: `found` (boolean), `answer` (string), `sources` (array)
2. **Per-source**: `memoryId` (non-empty string), `score` (finite number), `title` (string)
3. **Per-chunk** (if present): `chunkId` (non-empty string), `content` (string)

Invalid responses must be rejected with appropriate error handling.

The Backend must verify memory ownership before exposing results to the user.

## 17.4.1 Insufficient Memory Content Behavior

This section defines AI behavior when semantic retrieval finds relevant memories but those memories do not contain enough information to answer the user's question reliably.

### found Field Semantics

- **`found: false`** means NO relevant memories were found by semantic retrieval
- **`found: true`** means relevant memories WERE found, even if they are insufficient to fully answer the question

### Insufficient Content Handling

When relevant memories exist but do not contain enough information to answer reliably:

1. ✅ AI must return `found: true` (memories were found)
2. ✅ AI must return a clear `answer` indicating that available memories do not contain enough information
3. ✅ AI must NOT fabricate, hallucinate, or present unsupported information as fact
4. ✅ `sources` must contain only memories actually used/referenced in the answer
5. ✅ AI must ground the answer in what the memories DO contain, even if incomplete

**Example response when memories are insufficient:**

```json
{
  "found": true,
  "answer": "I found memories related to your question, but they don't contain enough information to answer reliably. The available memories mention [brief summary of what IS present], but more details would be needed to provide a complete answer.",
  "sources": [
    {
      "memoryId": "memory-id",
      "score": 0.87,
      "title": "Related memory title",
      "chunks": [...]
    }
  ]
}
```

### Answer Quality Requirements

The AI must:

- ✅ Be honest about information gaps
- ✅ Indicate when memories are insufficient
- ✅ Summarize what information IS available
- ✅ Avoid speculation or unsupported claims
- ❌ NOT fabricate details not present in memories
- ❌ NOT claim certainty when information is incomplete
- ❌ NOT return `found: false` when relevant memories exist

### Response Shape

The existing response shape remains unchanged:

```json
{
  "found": boolean,
  "answer": string,
  "sources": [...]
}
```

**No additional fields:**
- ❌ Do NOT add a `sufficient` field
- ❌ Do NOT add a `confidence` field
- ❌ Do NOT add any new response field

### Scope

This distinction applies only to AI Chat behavior (`POST /ai/chat/memories`).

It does NOT change:
- Semantic search behavior (`POST /ai/memories/search`)
- Memory processing behavior
- RAG retrieval logic

## 17.4.2 Answer-to-Source Linkage

This section defines how the AI answer is linked to source memories using the existing response structure.

### Canonical Linkage Mechanism

The existing `sources` array is the ONLY canonical mechanism for identifying memories used by the AI answer.

**No inline citations:**
- ❌ Do NOT introduce inline citation markers such as `[1]`, `[2]`, `[source 1]`, etc.
- ❌ Do NOT add citation IDs to the response
- ❌ Do NOT add source index fields
- ❌ Do NOT add new response fields for citation mapping

### sources Array Requirements

The `sources` array must contain ONLY memories actually used/referenced when generating the answer.

**Include in sources:**
- ✅ Memories that directly supported the answer content
- ✅ Memories that were referenced or cited in the answer
- ✅ Memories whose information appears in the answer

**Do NOT include in sources:**
- ❌ Memories retrieved as relevant but not actually used
- ❌ Memories that were semantically similar but not referenced
- ❌ Memories that could have been relevant but weren't used

### Answer Generation Guidelines

When generating the answer:

1. Use information from retrieved memories
2. Ground the answer in memory content
3. Include in `sources` only the memories actually used
4. Write a natural language answer without citation markers
5. Be clear about which memories support which parts of the answer through natural language

**Example of natural linkage:**

```json
{
  "found": true,
  "answer": "Based on your saved memories, you learned MongoDB performance optimization through indexing strategies. Your notes mention that compound indexes significantly improved query speed for user lookup operations.",
  "sources": [
    {
      "memoryId": "memory-123",
      "score": 0.94,
      "title": "MongoDB Performance Guide",
      "chunks": [...]
    }
  ]
}
```

The answer naturally references the source memory content without requiring `[1]` markers.

### Response Shape Preservation

The existing Chat response shape remains unchanged:

```json
{
  "found": true,
  "answer": "...",
  "sources": [...]
}
```

### Backend Responsibilities

The Backend remains responsible for:

- Validating the AI response structure
- Verifying memory ownership for all sources
- Mapping/enriching sources for the public Flutter response
- Constructing the final public API response per `API_CONTRACT.md`

### Public API Contract

The Flutter-facing API contract defined in `API_CONTRACT.md` remains unchanged.

This section defines only the internal AI → Backend contract.

---

## 17.5 Read-Only Constraint

Memory Chat is **RAG-only and read-only** for Sprint 1.

The AI Service must NOT:

- Create new memories
- Modify existing memories  
- Delete memories
- Mutate application source-of-truth data
- Interpret ordinary chat messages as save/search intents

Normal Chat answers questions about existing memories. It does not perform memory CRUD operations.

---

## 17.6 Explicit Save Note vs Normal Chat

These are separate operations with different endpoints:

**Normal Memory Chat:**
```text
User asks question
    ↓
POST /api/chat/memories (public API)
    ↓
Backend orchestration
    ↓
POST /ai/chat/memories (AI Service)
    ↓
Retrieval + RAG generation
    ↓
Answer + sources
    ↓
Backend validation
    ↓
Public response
```

**Explicit Save Note:**
```text
User explicitly selects "Save Note"
    ↓
POST /api/chat/memories/save-note (public API)
    ↓
Backend orchestration
    ↓
POST /ai/memories/prepare-note (AI Service)
    ↓
Title + content preparation
    ↓
Backend validation
    ↓
Backend creates Note
    ↓
Backend queues async processing
    ↓
POST /ai/memories/process (AI Service)
```

There is NO intent detection between these flows. The user's interaction with the application determines which operation occurs.

---

# 18. RAG / Memory Chat Boundary

Memovo Memory Chat is **RAG-only** for Sprint 1.

The purpose is to answer using the user's stored memories.

It is not a general-purpose assistant and does not mutate memories.

The architectural flow is:

```text
User question
    ↓
Backend (POST /api/chat/memories)
    ↓
AI RAG (POST /ai/chat/memories)
    ↓
Retrieval + answer generation
    ↓
Relevant memory context + answer
    ↓
Backend validation
    ↓
Public Chat response
```

The public Chat API is defined by the API contract.

The AI Service must not:

- create memories
- update memories
- delete memories
- modify application source-of-truth records

as a side effect of a RAG request.

---

# 19. Link / URL Boundary

This boundary is strict.

## Backend owns:

- URL validation
- SSRF protection
- DNS resolution
- IP validation
- redirect validation
- safe URL fetching
- page-content extraction
- metadata extraction
- canonical URL normalization
- persistence of extracted content and metadata

## AI Service does not:

- fetch URLs
- scrape pages
- crawl websites
- follow redirects
- dereference external URLs
- resolve external URLs for retrieval
- bypass Backend SSRF controls

If a Link contains:

```text
url = https://example.com/article
```

the AI Service must treat the supplied URL as data/context.

It must not independently access `example.com`.

The Backend must provide extracted content when AI processing requires page content.

---

# 20. Processing Freshness

The Backend controls memory freshness.

For asynchronous processing:

```text
Persist memory
    ↓
Create job
    ↓
Worker starts
    ↓
Reload latest memory
    ↓
Build AI request
    ↓
AI processing
```

For Links, stale extraction jobs are prevented from overwriting a newer URL state.

Deleted memories must not be resurrected by delayed AI processing.

If the Backend determines that the memory no longer exists or is no longer valid for processing, it must stop the processing flow.

---

# 21. Error Handling and Retry Policy

This section defines the canonical Backend ↔ AI Service error contract for all synchronous AI endpoints:

- `POST /ai/memories/prepare-note`
- `POST /ai/memories/process`
- `POST /ai/memories/search`
- `POST /ai/chat/memories`

## 21.1 Retry Ownership

**Backend owns retry behavior.**

The AI Service must NOT implement its own retry policy for Backend requests.

The Backend is solely responsible for:
- Detecting failures
- Classifying failures as retryable or non-retryable
- Implementing retry logic
- Applying exponential backoff
- Enforcing maximum retry attempts
- Timing out requests
- Recording failure state

## 21.2 Retryable Failures

The following failures should be retried by the Backend:

| Failure Type | HTTP Status | Retry? | Reason |
|---|---|:---:|---|
| Network/connection failure | N/A | ✅ Yes | Transient network issue |
| AI service unavailable | 503 | ✅ Yes | Service temporarily down |
| Request timeout | N/A | ✅ Yes | Transient timing issue |
| Too Many Requests | 429 | ✅ Yes | Rate limiting (see 21.2.1) |
| Internal Server Error | 500 | ✅ Yes | Transient server failure |
| Bad Gateway | 502 | ✅ Yes | Transient gateway issue |
| Gateway Timeout | 504 | ✅ Yes | Transient timeout |
| Other 5xx errors | 5xx | ✅ Yes | Server-side failure |

### 21.2.1 HTTP 429 Handling

When the AI Service returns HTTP 429 (Too Many Requests):

1. Treat as retryable when transient
2. If `Retry-After` header is provided:
   - Backend should respect it within configured retry/time limits
   - Do not wait indefinitely
   - Apply maximum wait time boundaries
3. If `Retry-After` is not provided:
   - Use exponential backoff per retry policy

## 21.3 Non-Retryable Failures

The following failures should NOT be retried:

| Failure Type | HTTP Status | Retry? | Reason |
|---|---|:---:|---|
| Bad Request | 400 | ❌ No | Backend payload error |
| Unauthorized | 401 | ❌ No | Authentication failure |
| Forbidden | 403 | ❌ No | Authorization failure |
| Not Found | 404 | ❌ No | Resource not found |
| Other 4xx errors | 4xx | ❌ No | Client/contract error |
| Invalid AI response | 200 | ❌ No | Schema/contract violation |
| Malformed output | 200 | ❌ No | Invalid generated content |

### Non-Retryable Rationale

- **4xx errors** indicate a deterministic contract or input error that will not be resolved by retrying
- **Invalid AI responses** (HTTP 200 with bad schema) indicate a contract violation, not a transient failure
- Retrying these failures wastes resources and delays failure detection

## 21.4 Retry Policy Parameters

The Backend must enforce the following retry parameters:

| Parameter | Value | Description |
|---|---|---|
| **Maximum attempts** | 3 | Total attempts including initial request |
| **Backoff strategy** | Exponential | Increasing delay between retries |
| **Timeout** | Backend-controlled | Per-request timeout (no indefinite waits) |
| **Maximum backoff** | Backend-controlled | Cap on exponential backoff delay |

**Exponential backoff example:**
```text
Attempt 1: immediate
Attempt 2: wait 1s
Attempt 3: wait 2s
```

The Backend may configure specific timeout and backoff values according to operational requirements.

## 21.5 Invalid AI Output Handling

When the AI Service returns HTTP success (200) but the response does NOT satisfy the canonical AI response schema:

1. ✅ Backend must detect the schema violation
2. ✅ Backend must treat it as a **contract violation**
3. ✅ Backend must **fail safely** (do not blindly retry)
4. ✅ Backend must log the invalid response for debugging
5. ❌ Backend must NOT retry immediately (infinite loop risk)
6. ❌ Backend must NOT accept/use the invalid output

**Invalid output examples:**
- Missing required fields (`found`, `results`, `answer`, etc.)
- Wrong field types (string instead of boolean, etc.)
- Malformed structure (not an object, not an array, etc.)
- Invalid embedding dimensions (not 1024)
- Non-finite numbers (NaN, Infinity)

## 21.6 Endpoint-Specific Failure Behavior

### POST /ai/memories/prepare-note

**Success:** Backend continues normal Note creation flow

**Retryable failure:** Retry according to policy (max 3 attempts)

**Non-retryable failure or invalid output:**
- Fail safely
- Do NOT create an incomplete or invalid Note
- Return appropriate error to client per public API contract

The Backend must not create a malformed Note because AI preparation failed.

### POST /ai/chat/memories

**Success:** Backend maps/validates and returns public Chat response

**Retryable failure:** Retry according to policy (max 3 attempts)

**Non-retryable failure or invalid output:**
- Fail safely
- Do NOT fabricate an answer
- Do NOT substitute a Backend-generated answer
- Return appropriate error to client per public API contract

The Backend must never fabricate or substitute an AI answer after generation failure.

### POST /ai/memories/process

**Success:** Backend validates and persists chunks/embeddings

**Retryable failure:** Retry according to policy (max 3 attempts)

**Non-retryable failure or invalid output:**
- Fail safely
- Mark processing job as failed (BullMQ)
- Do NOT persist invalid vectors
- Apply BullMQ retry policy for job-level retries

### POST /ai/memories/search

**Success:** Backend validates and maps search results

**Retryable failure:** Retry according to policy (max 3 attempts)

**Non-retryable failure or invalid output:**
- Fail safely
- Return appropriate error to client per public API contract
- Do NOT return malformed search results

## 21.7 Public API Boundary

This section defines **only the internal Backend ↔ AI Service behavior.**

**Do NOT:**
- ❌ Introduce new Flutter-facing error fields
- ❌ Change public API error responses
- ❌ Add new error endpoints
- ❌ Modify public API response structures

**Existing API_CONTRACT.md remains the source of truth for public API errors.**

The Backend maps internal AI failures to appropriate public API error responses as defined in the API contract.

## 21.8 Preserved Contracts

This error handling policy preserves all previously finalized decisions:

- ✅ No `whySaved` field
- ✅ Link `source` contains only four AI-facing nullable fields
- ✅ `extractedContent` remains optional/nullable/empty-allowed
- ✅ Search AI request remains `{userId, query}`
- ✅ Backend owns memory type filtering
- ✅ Chat history limits from v1.8 Section 17.2.1
- ✅ Backend-owned `conversationId` from v1.8 Section 17.2.2
- ✅ `found: true` may mean retrieved memories are insufficient (v1.8 Section 17.4.1)
- ✅ Sources-only answer/source linkage (v1.8 Section 17.4.2)
- ✅ AI remains stateless
- ✅ RAG-only/read-only Chat

## 21.9 Approved Addendum: Malformed Model Output Detected by the AI Service

Backend clarification (Point 10, 2026-09-12), synchronized here from the implementation plan so the canonical file carries it.

When the AI Service detects malformed model/generated output or an output-schema violation **before** responding, it returns:

```text
HTTP 502 Bad Gateway
error.code = AI_INVALID_RESPONSE
error.retryable = false
```

- This is **always** non-retryable: an explicit exception to the retryable 5xx rows in 21.2, including the generic 502 row.
- The AI Service never intentionally emits an invalid HTTP 200 body to signal failure.
- The canonical classification is **HTTP status + error.code**. `error.retryable` is consistent metadata and never an override.
- The Backend also does not retry an HTTP-200 body that fails canonical schema validation (21.5).

### AI Service error codes (informative)

The envelope is `{"error": {"code", "message", "retryable"}}`. `Retry-After` is sent only with `RATE_LIMITED`, bounded, and only when the upstream provider supplied one.

| HTTP | `error.code` | `retryable` | Meaning |
|---|---|:---:|---|
| 422 | `INVALID_INPUT` | false | Request failed schema/limit validation |
| 400 | `INVALID_REQUEST` | false | Transport-level client error (404/405 keep their status) |
| 429 | `RATE_LIMITED` | true | Generation provider rate limited |
| 500 | `INTERNAL_ERROR` | true | Service defect; retried under the 5xx rule |
| 500 | `CHUNKING_FAILED` | true | Chunking defect; retried under the 5xx rule |
| 502 | `AI_INVALID_RESPONSE` | **false** | Malformed/truncated model output or output-schema violation |
| 503 | `EMBEDDING_FAILED` | true | Embedding inference failed |
| 503 | `VECTOR_SEARCH_FAILED` | true | Vector engine unavailable |
| 503 | `MODEL_UNAVAILABLE` | true | Embedding model not loaded |
| 503 | `GENERATION_UNAVAILABLE` | true | Generation disabled, unconfigured, or provider unreachable/upstream failure |
| 504 | `TIMEOUT` | true | Vector search or generation deadline exceeded |

---

# 22. Response Validation

AI output is untrusted external service output from the Backend's perspective.

The Backend must validate:

- expected response structure
- `memoryId`
- chunk collection
- `chunkId`
- `chunkIndex`
- chunk content
- embedding type
- embedding dimension
- numeric embedding values

Invalid AI output must not be persisted as vector source data.

---

# 23. Data Ownership

| Responsibility | Backend | AI Service |
|---|:---:|:---:|
| Public API | Yes | No |
| Authentication | Yes | No |
| Authorization | Yes | No |
| MongoDB application records | Yes | No |
| BullMQ | Yes | No |
| Worker lifecycle | Yes | No |
| Retry/backoff | Yes | No |
| Link fetching | Yes | No |
| Link extraction | Yes | No |
| SSRF protection | Yes | No |
| AI processing | Orchestrates | Yes |
| Chunk generation | No | Yes |
| Embeddings | No | Yes |
| Query embedding | No | Yes |
| Vector retrieval | Backend orchestrates/validates | AI Service performs |
| Vector persistence | Yes | No |
| Vector reconciliation | Yes | No |
| Vector cleanup | Yes | No |
| RAG generation | Orchestrates | Yes |
| Memory mutation | Yes | No |
| Chat Save Note preparation | Orchestrates | Yes |

---

# 24. Security Boundaries

The Backend remains the trust boundary for application data.

The AI Service must not be treated as an authority capable of:

- changing ownership
- changing user identity
- mutating memory records
- bypassing authorization
- directly modifying application source-of-truth data

The Backend must validate AI output before using it.

User identity must originate from the authenticated Backend context.

External URL access must remain exclusively under Backend-controlled SSRF-safe processing.

---

# 25. Contract Rules

The following rules are mandatory:

1. AI Service is synchronous HTTP-only.
2. Backend owns BullMQ and job lifecycle.
3. Backend owns retry and backoff behavior.
4. AI Service does not perform intent detection for save/search operations.
5. Chat Save Note preparation uses `POST /ai/memories/prepare-note`.
6. Memory processing uses `POST /ai/memories/process`.
7. Semantic memory retrieval uses `POST /ai/memories/search`.
8. Memory Chat RAG generation uses `POST /ai/chat/memories`.
9. Backend owns external URL fetching and content extraction.
10. AI Service never fetches, scrapes, crawls, or dereferences external URLs.
11. AI-generated embeddings are 1024-dimensional.
12. Backend validates AI responses before persistence/use.
13. Backend owns vector persistence and reconciliation.
14. AI Service does not own the application's MongoDB source of truth.
15. Sprint 1 Memory Chat is RAG-only and read-only.
16. Normal Chat does not mutate memories.
17. Save Note is a separate explicit operation.
18. Public API behavior is governed by `API_CONTRACT.md`.

---

# 26. Related Contracts

This document defines only the Backend ↔ AI Service boundary.

For the public Memovo API contract, see:

`docs/contracts/API_CONTRACT.md`

For system architecture and operational behavior, refer to the corresponding documentation under:

- `docs/architecture/`
- `docs/data/`
- `docs/security/`
- `docs/operations/`

These documents must reference this contract rather than creating competing AI contract definitions.

---

# 27. Version History

## 27.0 Addendum to Version 1.9 (2026-09-12)

- Added Section 21.9: malformed model output detected inside the AI Service returns `502 AI_INVALID_RESPONSE` with `retryable: false`, always non-retryable; status + code classification takes precedence over `retryable`.
- Recorded the AI Service error-code table (informative) and the consistency review of `INTERNAL_ERROR` / `CHUNKING_FAILED` metadata under the 5xx rule.

## 27.1 Version 1.9 (2026-09-07)

**Points 7-10: conversationId, Insufficient Memories, Answer-Source Linkage, Error Handling**

### Point 7: Canonical conversationId Behavior (Section 17.2.2)

- Formalized `conversationId` as Backend-owned conversation identifier
- Documented conversation ownership:
  - Backend owns conversation state and persistence
  - AI Service is stateless with respect to conversations
  - AI Service must NOT store conversation/session state
- Documented conversation flow:
  - First request: Backend creates conversationId, returns to client
  - Follow-up requests: Client sends conversationId, Backend loads history
  - Backend applies history limits before sending to AI
  - Backend sends bounded history on each request
- Clarified AI Service conversationId usage:
  - May receive as identifier (logging/debugging/correlation)
  - Must NOT use it to retrieve conversation state
  - Must NOT use it to persist conversation state
  - Does NOT own conversation lifecycle
- Documented conversation ownership enforcement:
  - conversationId bound to specific user
  - Backend validates ownership
  - Must never allow access to another user's conversation
- Clarified conversationId scope:
  - Used for conversation continuity and history retrieval
  - Does NOT alter RAG retrieval, semantic search, or memory processing
- Added architecture diagram showing Backend provides all context via history array

### Point 8: Memories Found but Not Sufficient (Section 17.4.1)

- Formalized `found` field semantics:
  - `found: false` = no relevant memories found
  - `found: true` = relevant memories found, even if insufficient
- Documented insufficient content handling requirements:
  - Return `found: true` when memories exist but are insufficient
  - Provide clear answer indicating information gaps
  - Must NOT fabricate, hallucinate, or present unsupported claims
  - `sources` must contain only memories actually used/referenced
  - Ground answer in available memory content, even if incomplete
- Added example response for insufficient memories
- Documented answer quality requirements:
  - Be honest about information gaps
  - Indicate when memories are insufficient
  - Summarize available information
  - Avoid speculation or unsupported claims
- Clarified NO additional response fields (no `sufficient`, `confidence`, etc.)
- Preserved existing response shape: `{found, answer, sources}`
- Scope: applies only to Chat behavior, not semantic search

### Point 9: Answer-to-Source Linkage (Section 17.4.2)

- Formalized `sources` array as canonical linkage mechanism
- Explicitly prohibited inline citations:
  - NO citation markers like `[1]`, `[2]`, `[source 1]`
  - NO citation IDs
  - NO source index fields
  - NO new response fields for citation mapping
- Documented sources array requirements:
  - Include ONLY memories actually used/referenced in answer
  - Do NOT include memories retrieved but not used
  - Do NOT include semantically similar but unreferenced memories
- Provided answer generation guidelines:
  - Use information from retrieved memories
  - Ground answer in memory content
  - Include in sources only memories actually used
  - Write natural language answer without citation markers
- Added example showing natural linkage without markers
- Preserved existing response shape: `{found, answer, sources}`
- Backend remains responsible for validation, mapping, enrichment
- Public API contract unchanged

### Point 10: Canonical AI Error Handling & Retry Policy (Section 21)

- Complete rewrite of Section 21 with comprehensive error/retry contract
- Formalized retry ownership: Backend owns all retry behavior
- Documented retryable failures:
  - Network/connection failures
  - AI service unavailable (503)
  - Request timeout
  - HTTP 429 (Too Many Requests) with Retry-After handling
  - HTTP 5xx server-side failures
- Documented non-retryable failures:
  - HTTP 4xx client/contract errors (400, 401, 403, 404)
  - Invalid AI responses (HTTP 200 with schema violation)
  - Malformed output
- Established retry policy parameters:
  - Maximum 3 total attempts (including initial)
  - Exponential backoff strategy
  - Backend-controlled timeout
  - No indefinite retries
- Documented invalid AI output handling:
  - Detect schema violations
  - Treat as contract violations, not transient failures
  - Fail safely, do not blindly retry
  - Log for debugging
- Documented endpoint-specific failure behavior:
  - `/ai/memories/prepare-note`: Do NOT create incomplete Note
  - `/ai/chat/memories`: Do NOT fabricate/substitute answer
  - `/ai/memories/process`: Mark job failed, do NOT persist invalid vectors
  - `/ai/memories/search`: Do NOT return malformed results
- Clarified public API boundary:
  - Internal Backend ↔ AI contract only
  - No new Flutter-facing error fields
  - API_CONTRACT.md remains source of truth for public errors
- Listed all preserved contracts (whySaved, extractedContent, type filtering, history limits, conversationId, found semantics, sources linkage, stateless AI, RAG-only)

## 27.2 Version 1.8 (2026-09-07)

**Canonical Chat History Contract (`POST /ai/chat/memories`):**

- Formalized chat history contract in Section 17.2.1
- Documented history structure:
  - Each message contains `role` and `content`
  - Allowed roles: `user`, `assistant` only
  - Forbidden roles: `system` and any other roles
- Established size limits:
  - Maximum messages: 30
  - Maximum per-message content size: 6,000 characters
  - Maximum total content size: 30,000 characters
- Documented Backend history responsibilities:
  - Validation of all messages
  - Role validation (only `user`/`assistant` allowed)
  - Per-message size enforcement
  - Total size enforcement
  - Message count enforcement
  - Overflow trimming (oldest messages removed first)
  - Message integrity (no partial truncation)
- Documented history trimming behavior:
  - Remove oldest messages first when limits exceeded
  - Keep newest messages that fit within both limits
  - Messages must be included in full or excluded entirely
- Clarified AI Service history expectations:
  - Receives already-bounded history from Backend
  - Must NOT perform its own history trimming
  - Must NOT enforce size limits
  - Does NOT own conversation state
- Documented conversation state ownership:
  - Backend owns conversation state and history management
  - AI Service is stateless
  - `conversationId` used for continuity, not lifecycle management
- Clarified that limits are safety/context boundaries:
  - Backend NOT required to send maximum 30 messages on every request
  - Backend should send only relevant context within limits
- Added valid history examples (no history, short history, omitted)

## 27.3 Version 1.7 (2026-09-07)

**Memory Type Filter Contract Clarification:**

- Formalized that `type`/`memoryType` is NOT sent from Backend to AI Service
- Updated Section 14 with explicit search request schema:
  - Required fields: `userId`, `query`
  - Fields NOT sent: `type`/`memoryType`, `limit`
- Added Section 14.2 documenting memory type filtering architecture:
  - Type filtering is exclusively a Backend responsibility
  - AI performs unified semantic search across all memory types
  - Backend applies type filtering AFTER receiving AI response
  - Backend uses MongoDB classification queries to determine memory types
- Added Section 16.5 documenting Backend type filtering and limit application:
  - Detailed 8-step Backend processing flow
  - Type filter scenarios table (omitted/`note`/`link`)
  - Limit application occurs AFTER type filtering
  - Complete example flow from public API → AI → Backend → public API
- Clarified that when public API omits `type`, Backend returns both Notes and Links
- Reinforced that AI Service does NOT receive or apply memory-type filters
- Reinforced that AI Service does NOT receive or apply limit parameter

## 27.4 Version 1.6 (2026-09-07)

**Canonical Search Response Schema (`POST /ai/memories/search`):**

- Documented the exact internal AI → Backend response contract in Section 16
- Formalized response schema with explicit field requirements:
  - Top-level: `found` (boolean), `results` (array)
  - Per-result: `memoryId`, `score`, `title`, `tags`, `chunks`
  - Per-chunk: `chunkId`, `content`
  - No-match response: `{found: false, results: []}`
- Clarified this is an INTERNAL response, not the public Flutter-facing API
- Documented field exclusions (AI must NOT return):
  - Backend-mapped fields: `id`, `matchedChunks`, `similarityScore`
  - MongoDB-fetched fields: `type`, `content`, `url`
  - Application metadata: `createdAt`, `updatedAt`, `userId`
- Documented Backend response mapping responsibilities:
  - Validation, deduplication, type filtering (Backend-side)
  - MongoDB enrichment, field renaming, ownership verification
  - Public API response construction per API_CONTRACT.md
- Added comprehensive response validation requirements
- Updated Section 15 to clarify:
  - AI does NOT receive or apply memory-type filters
  - Type filtering is Backend responsibility (post-AI-response)
  - AI performs unified semantic search across all memory types

## 27.5 Version 1.5 (2026-09-07)

**Legacy Field Removal: `whySaved`**

- Explicitly stated that `whySaved` is **permanently removed** from Sprint 1
- `whySaved` is NOT part of any Backend → AI request or AI processing input
- Added Section 8.5 documenting the removal scope and contract enforcement
- Clarified that AI Service must NOT:
  - Accept `whySaved` as a processing input field
  - Include `whySaved` in embeddings
  - Use `whySaved` in AI context generation
  - Map `whySaved` to another field automatically
- Documented current Link context fields:
  - `content` = user-provided context (the canonical user input field)
  - `extractedContent` = Backend-extracted page content
  - `source.sourceDescription` = Backend-extracted source metadata
- Clarified that `source.sourceDescription` is NOT a replacement for `whySaved`
- Added contract enforcement guidelines for any legacy references

## 27.6 Version 1.4 (2026-09-07)

**Processing Request Schema Formalization (`/ai/memories/process`):**

- Formalized Note processing request schema with explicit field constraints:
  - `type`: required, value must be `"note"`
  - `title`: required, 0–200 characters
  - `content`: required, 1–10,000 characters
  - `tags`: optional, nullable, maximum 20 tags, maximum 50 characters per tag
  
- Formalized Link processing request schema with explicit field constraints:
  - `type`: required, value must be `"link"`
  - `url`: required, maximum 2048 characters
  - `title`: required, 1–500 characters
  - `content`: optional, nullable, empty string allowed, maximum 1000 characters
  - `tags`: optional, nullable, maximum 20 tags, maximum 50 characters per tag
  - `source`: required object (individual fields may be null)
  - `extractedContent`: optional, nullable, empty string allowed

- Clarified Link `source` object structure sent to AI Service:
  - AI receives ONLY: `sourceTitle`, `sourceDescription`, `authorName`, `publicationDate`
  - AI does NOT receive: `platform`, `contentType`, `thumbnailUrl`, `canonicalUrl`
  - These excluded fields are Backend-only metadata

- Clarified `extractedContent` architectural responsibilities:
  - Backend-generated extracted page content
  - Backend performs URL fetch and content extraction
  - AI Service receives it when available
  - AI Service does NOT fetch URLs itself
  - Optional, nullable, empty string allowed
  - No new AI-specific character limit; Backend controls extraction/storage/payload limits

- Distinguished field purposes:
  - `content` = user-provided Link context
  - `extractedContent` = Backend-extracted page content
  - `source.*` = Backend-extracted selected metadata

- Reinforced SSRF boundary: URL field is contextual information, NOT an instruction for AI to fetch resources

## 27.7 Version 1.3

- Corrected Normal Memory Chat routing to `POST /ai/chat/memories`.
- Clarified that `POST /ai/memories/search` is retrieval-only.
- Clarified that AI Service performs Atlas Vector Search while Backend orchestrates/validates and owns vector persistence/reconciliation/cleanup.
- Kept explicit Chat Save Note preparation on `POST /ai/memories/prepare-note`.

---

# 28. Source of Truth

This file is the canonical documentation source for the **Backend ↔ AI Service contract**.

Changes to:

- AI endpoints, including Chat Save Note preparation and Memory Chat RAG
- request/response schemas
- AI responsibilities
- Backend/AI ownership boundaries
- embedding dimensions/model
- processing/search/chat semantics
- URL-fetching responsibility

must be reflected here when the implementation changes.

This document must not be treated as a substitute for the actual implementation. When implementation and documentation diverge, the discrepancy must be identified and resolved explicitly.
