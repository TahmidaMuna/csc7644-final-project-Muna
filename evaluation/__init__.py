"""
evaluation package
------------------
Scripts for evaluating the CSR Allocation Agent against historical
Louisiana disaster events with documented damage assessments.

Modules
-------
eval_runner
    Runs the agent on Hurricane Ida (DR-4611) and the August 2016 floods
    (DR-4277), computes Precision@5 against ground-truth parish rankings,
    and prints a formatted summary with narrative quality rubric.
"""
