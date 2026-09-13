# Resume prompt for Claude Code

Continue the Memovo AI work from the CURRENT working tree. Do not restart P0 or overwrite work you already completed after it.

Read docs/AI_CONTRACT_FINAL.md and docs/SPRINT2_GENERATION_PLAN.md. The plan is now a clean English document and the only implementation plan. D0_DECISIONS.md and the outdated Arabic integration guide were removed intentionally; do not recreate them or follow their old alternatives. Backend Points 1–10 are finalized, including AI_INVALID_RESPONSE in plan section 4. Do not ask those questions again. The owner's adoption of the new contract supersedes old sibling-document/CLAUDE.md prose requiring legacy fields or prohibiting this authorized generation work.

First inspect git status/diff and the actual source/tests. The latest review found P1/P2 partially implemented: Note/Link schemas, source fields, canonical content, process/search response mappings, Atlas defaults and evaluation fixtures have edits, while many contract/integration tests still use description/whySaved or NO_MATCH_MESSAGE. Treat this as ongoing migration, not completed work and not a reason to revert it. Recheck because you may have progressed since that snapshot.

Resume by finishing those migrations and their tests. Preserve correct new code and P0 guard/correlation/AiServiceError logging. Run relevant checks, then proceed through P3–P8 in the plan in reviewable phases. Do not stop after writing another plan or merely acknowledging readiness.

Critical decisions:

- Exact finalized Note/Link limits and required-nullable four-field source; whySaved removed without mapping; extractedContent has no invented AI character cap.
- Process returns memoryId/chunks with chunkIndex/1024 embeddings. Search returns found/results, search chunks without chunkIndex, no-match without message. No AI type filter/public limit or semantic retrieval change for wire migration.
- History is 30 messages, 6000 characters each, 30000 total, user/assistant roles. Backend trims COMPLETE old messages; AI does NOT trim history. conversationId is an identifier, not AI session state or a retrieval selector.
- Chat returns found/answer/sources; insufficient evidence still found=true. Used sources only; no inline citations or new fields. Prepare-note returns title/content only, no tags or persistence.
- Malformed model output detected inside AI: HTTP 502, AI_INVALID_RESPONSE, retryable:false. Status+code classification takes precedence over retryable metadata. This is an explicit non-retryable exception to service/transient 5xx; invalid HTTP-200 output at Backend also must not be retried. No public Flutter API change.
- Backend owns max 3 TOTAL attempts/backoff/Retry-After. Disable AI/SDK automatic retries.
- Generation: OpenRouter Hosted API, exact nvidia/nemotron-3-super-120b-a12b:free. No Llama deployment, generator weight downloads, paid/alternate model fallback or openrouter/free. Embeddings remain unchanged.

Implement the prepared env settings listed in plan section 5. The owner adds MEMOVO_OPENROUTER_API_KEY locally. Do not print .env/keys, overwrite existing secrets, or commit them. Keep generation disabled until configured/verified; process/search must not require the key. Verify endpoint availability/capabilities with a synthetic opt-in smoke test when access is available. Lack of access blocks real inference evidence only, not fake-provider/local implementation.

Preserve threshold 0.75 during contract migration. Score alignment and live Atlas evaluation are explicit follow-ups, not permission to silently change retrieval. Backend owns Atlas creation, data writes and reconciliation; AI remains read-only.

For each phase run focused tests and pytest/Ruff check/Ruff format check/mypy. No model downloads or external calls in ordinary CI. Report actual results and distinguish implemented, locally tested and externally verified. Restore no obsolete deployment guide; write an accurate current guide at the handoff phase and update stale references intentionally.

Proceed autonomously through authorized independent work. Only ask for a genuinely missing decision/access that blocks a dependent action; do not repeat finalized questions or stop all work for an unavailable external service. Do not commit, push, deploy, provision/write Atlas, add unrelated features or change public API contracts. Leave all changes reviewable and uncommitted.
