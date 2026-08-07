"""
analysis/

Shared analysis utilities. `embeddings` provides the sentence-transformer
model used by the validation scripts and by symptom_scoring; `validation/`
holds the manipulation-validity checks.

Kept import-free at package level so `import analysis.embeddings` stays cheap.
"""
