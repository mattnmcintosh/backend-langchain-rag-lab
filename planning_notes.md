# Planning Notes: LangChain RAG API

Complete each section before submitting. Replace every TODO with your own notes.

## 1. System Goal

- User or role: On-call developer responding to production incidents
- Business problem: SWEs should have high quality, immediate, and accurate guidance during production incidents to assist resolution. 
                    General models can be too vague or not specialized enough with company policy without a supporting architecture to be able to maintain the required amount of guidance.
- Approved knowledge source: Internal reliability runbooks covering checkout API errors, rollback coordination, status page updates, exposed API keys, rate limits, and delayed data exports.
- Endpoint route: POST /api/ask

## 2. Manual RAG Workflow Map

Name the manual RAG steps this LangChain version is organizing.

- Receive question: Extract and validate the incoming incident query string from the JSON payload.
- Retrieve context: Query the Chroma vector store for the top-k most relevant internal runbook chunks using semantic search.
- Build prompt: Combine the user's question with formatted runbook content inside a structured prompt template.
- Call model: Send the populated prompt to the local LLM model client via LangChain orchestration.
- Parse or format output: Extract the generated answer text and strip unwanted conversational padding.
- Return sources: Attach structured source metadata (runbook ID, title, section, chunk ID) to provide clear attribution.
- Verify quality: Ensure the response relies strictly on retrieved context and safely triggers a fallback if context chunks are missing or irrelevant.

## 3. LangChain Component Mapping

Map the manual workflow to LangChain-supported pieces.

- Prompt string maps to: ChatPromptTemplate or PromptTemplate with designated input variables ({context}, {question})
- Chroma search logic maps to: Chroma vector store wrapper utilizing .as_retriever().
- Direct model call maps to: Chat model integrations like ChatOllama or ChatOpenAI.
- Function-to-function workflow maps to: LangChain Expression Language (LCEL) chains using the pipe operator (|).
- Manual output cleanup maps to: StrOutputParser() to extract clean text strings automatically.
- Manual testing maps to: Pytest mock fixtures, runnable unit tests, and LangChain debugging/tracing tools (-v or callback handlers).

## 4. Response Contract

List the fields a successful response should include.

- answer (string): The grounded, concise resolution steps or guidance pulled from the approved runbooks.
- sources (list of objects): Metadata dictionaries containing unique source IDs, titles, sections, and chunk IDs for auditability.
- debug / metadata (optional dictionary): LangChain execution steps or traceability metadata to aid in troubleshooting.

## 5. Fallback Behavior

Explain what the system should do when no approved context is available.

If the vector store returns no relevant runbook chunks or if the similarity score falls below threshold limits, the system must bypass the LLM entirely and return a safe fallback message stating that approved runbooks do not contain enough information to resolve the incident safely, directing the engineer to escalate to a senior team lead or incident commander.

## 6. Verification Plan

List at least four checks you should run before trusting the refactor.

- Verify that invalid inputs (blank questions, missing JSON bodies) correctly return a 400 status code without triggering the retriever or model.
- Verify that a valid question successfully traverses the LCEL chain, returning a 200 status code with both an answer and source metadata.
- Verify that zero-context queries correctly short-circuit to the fallback response rather than hallucinating an answer.
- Verify that model service outages or connection failures cleanly map to a 502 error response.

## 7. Reflection

Explain what LangChain simplified and what may be harder to inspect.

LangChain significantly simplified boilerplate wiring by replacing manual string formatting and separate client calls with clean declarative pipelines (LCEL). However, because execution flows through abstracted runnables and chains, debugging hidden error states or inspecting exact intermediate prompts can sometimes feel less transparent than explicit procedural Python code.
