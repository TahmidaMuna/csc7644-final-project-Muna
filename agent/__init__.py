"""
agent package
-------------
Contains the LLM agent, tool definitions, and prompt templates
for the Post-Disaster CSR Fund Allocation system.

Public API
----------
CSRAllocationAgent
    Primary class for running the agentic CSR analysis loop.
    Accepts a FEMA disaster number and returns a plain-text allocation report.
"""

from agent.llm_agent import CSRAllocationAgent

__all__ = ["CSRAllocationAgent"]
