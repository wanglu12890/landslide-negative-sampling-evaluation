# Public release audit

## Status

Offline release candidate only; **not yet an authorized open-source release**. No commit, push, repository creation or remote upload was performed. Code ownership and an explicit license, derivative-ID/result permissions, and factor metadata require author confirmation.

## Included / excluded

|Class|Handling|Remaining issue|
|---|---|---|
|Original implementation|Necessary four sampling/model packages only, source hashes retained|Authors must confirm ownership and any external code origins.|
|Sample IDs|Saved IDs, class labels and saved order only; run-qualified|Confirm permission to publish derivative identifiers; not globally spatial IDs.|
|Results|Aggregate copied numeric tables and code-version records|Confirm derivative-data/result publication rights.|
|Coordinates and features|Not included in sample/event files|Grid origin and CRS remain as coarse methodological parameters already described in manuscript.|
|Imagery/geographic datasets|Excluded|Original data licenses unresolved.|
|Models/predictions/large caches|Excluded|Provide separately only if authorized.|
|Logs/private metadata|Raw logs/XML excluded; reduced run status/metrics/version records retained|Reduced summaries are extracted records, not complete logs.|
|Credentials and local connection info|Pattern scan of text files, extension/size checks|A pattern scan is not a guarantee of absence; author review remains necessary.|
|Local absolute paths|Structured config paths adapted; copied result run/raster columns removed|Relative original run provenance retained for traceability.|
|Third-party dependencies|Requirements references only; no vendored dependency implementations|Dependencies retain their own licenses.|
|License|No public grant until author decision|Do not label the current tree MIT/Apache or claim OSI licensing.|

See PUBLIC_RELEASE_SCAN.json for the executed scan findings, VALIDATION.json for isolated execution checks, and PAPER_CONSISTENCY.md for the manuscript comparison. No complete private factor metadata XML is copied because it contains workstation/network identifiers and does not establish acquisition dates.
