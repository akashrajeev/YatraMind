# YatraMind Refactoring Roadmap

The refactor is being performed on a dedicated branch and merged through a draft PR. The guiding rule is to preserve validated operational behavior while removing duplicated architecture.

## Completed

- Trustworthy CI for backend tests and frontend typecheck/build.
- Runtime/generated simulation artifacts removed from source control and ignored.
- Historical optimization regression contract added.
- Fleet-planning legacy compatibility restored.
- Canonical trainset and optimization domain types introduced.
- Structured Tier-1 safety constraints introduced.
- Pure Tier-2/Tier-3 scoring introduced.
- Canonical optimization engine introduced with OR-Tools and deterministic fallback.
- Legacy monolithic optimizer reduced to a compatibility facade.
- Primary optimization API migrated to the application-service/repository path.
- Trainset and optimization-history repository boundaries introduced.
- Deterministic ML risk-provider boundary introduced.
- Explainability moved behind a non-critical service boundary.
- Optimization and assignments pages moved into feature modules without route changes.
- Legacy API-key path changed to fail closed when unconfigured.
- Stabling implementation isolated behind a compatibility adapter while preserving the original algorithm.

## Remaining

- Remove remaining direct database access from legacy endpoints.
- Consolidate the legacy rule engine and role-assignment solver where no longer needed.
- Decompose result construction and remaining legacy API adapters.
- Configure a supported frontend test runner instead of excluding legacy Jest tests.
- Remove obsolete setup/debug/output files after reference audit.
- Reduce duplicate database implementations to adapters behind repository interfaces.
- Standardize remaining dependency/configuration deprecations.
- Complete authentication/security audit and production hardening.
- Finish frontend decomposition of the remaining large pages/components.

## Verification gates

1. CI green.
2. Canonical optimizer unit/regression tests green.
3. API contract tests green.
4. Frontend typecheck/build green.
5. No tracked runtime/generated artifacts.
6. No active legacy optimizer path for the primary optimization path.
7. No safety decision based on human-readable reason strings.
8. Main branch remains behavior-compatible until the migration is explicitly approved.
