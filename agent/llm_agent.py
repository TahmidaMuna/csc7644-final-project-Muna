"""
llm_agent.py
------------
Implements the agentic loop for the Post-Disaster CSR Fund Allocation Agent.

The agent uses OpenAI GPT-4o with function calling to:
    1. Identify affected parishes via FEMA API
    2. Retrieve NRI, SVI, and Census data for each parish
    3. Augment reasoning with RAG-retrieved methodology context
    4. Rank parishes and generate a plain-language allocation report

Usage:
    agent = CSRAllocationAgent(openai_api_key="sk-...")
    report = agent.run("Allocate CSR funds for disaster 4611")
"""

import json
import os
import time
from typing import Optional

from openai import OpenAI, RateLimitError

from agent.tools import TOOL_DEFINITIONS, dispatch_tool
from agent.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLE
from rag.retriever import RAGRetriever, retrieve_context


# Maximum number of tool-call rounds before forcing a final answer
MAX_TOOL_ROUNDS = 20

PRIMARY_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
TOKEN_WARNING_THRESHOLD = 25_000


def compact_tool_result(tool_name: str, raw: dict) -> dict:
    """
    Keep only the fields needed for scoring and narrative generation.
    """
    if "error" in raw:
        return {"error": raw["error"]}

    if tool_name == "nri_lookup":
        return {
            "parish": raw.get("parish"),
            "fips": raw.get("fips"),
            "risk_score": raw.get("risk_score"),
            "risk_rating": raw.get("risk_rating"),
            "eal_usd": raw.get("expected_annual_loss_usd"),
            "eal_score": raw.get("expected_annual_loss_score"),
            "hurricane_eal_usd": raw.get("hurricane_eal_usd"),
            "hurricane_eal_score": raw.get("hurricane_eal_score"),
            "flood_eal_usd": raw.get("inland_flood_eal_usd"),
            "coastal_flood_eal_usd": raw.get("coastal_flood_eal_usd"),
        }
    if tool_name == "svi_lookup":
        return {
            "parish": raw.get("parish"),
            "fips": raw.get("fips"),
            "overall_vulnerability": raw.get("overall_svi_percentile"),
            "theme1_socioeconomic": raw.get("theme1_socioeconomic"),
            "theme2_household": raw.get("theme2_household_characteristics"),
            "theme3_minority": raw.get("theme3_racial_ethnic_minority"),
            "theme4_housing": raw.get("theme4_housing_transportation"),
        }
    if tool_name == "census_lookup":
        return {
            "parish": raw.get("parish"),
            "fips": raw.get("fips"),
            "median_income": raw.get("median_household_income_usd"),
            "poverty_rate": raw.get("poverty_rate_pct"),
            "uninsured_count": raw.get("uninsured_count"),
            "housing_burden": raw.get("housing_cost_burden_pct"),
            "total_population": raw.get("total_population"),
        }
    if tool_name == "get_affected_parishes":
        return {
            "disaster_number": raw.get("disaster_number"),
            "disaster_title": raw.get("disaster_title"),
            "incident_type": raw.get("incident_type"),
            "parishes": raw.get("parishes", []),
        }
    return raw


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Rough token estimate for rate-limit guardrails."""
    return sum(len(str(message.get("content", ""))) for message in messages) // 4


class CSRAllocationAgent:
    """
    Agentic LLM wrapper for post-disaster CSR fund allocation analysis.

    Attributes
    ----------
    client : OpenAI
        Authenticated OpenAI client.
    rag : RAGRetriever
        Vector store retriever loaded from the methodology corpus.
    verbose : bool
        If True, print tool call details to stdout during execution.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        corpus_dir: str = "data/corpus",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the agent with an OpenAI client and RAG retriever.

        Parameters
        ----------
        openai_api_key : str, optional
            OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
        corpus_dir : str
            Path to the directory containing methodology PDF files for RAG.
        verbose : bool
            Print tool names and result previews during agentic execution.
        """
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key to CSRAllocationAgent()."
            )
        self.client = OpenAI(api_key=api_key)
        self.corpus_dir = corpus_dir
        self.rag = RAGRetriever(corpus_dir=corpus_dir, openai_api_key=api_key)
        self.verbose = verbose

    def _build_initial_messages(self, user_query: str) -> list[dict]:
        """
        Construct the initial message list with system prompt and user query.

        Injects the few-shot example into the system message. RAG context is
        added once later in the loop after data retrieval starts.

        Parameters
        ----------
        user_query : str
            The user's natural language request.

        Returns
        -------
        list[dict]
            OpenAI-formatted message list ready for a completion call.
        """
        system_content = SYSTEM_PROMPT
        system_content += "\n\n" + FEW_SHOT_EXAMPLE

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

    def run(self, user_query: str) -> str:
        """
        Execute the full agentic loop for a given CSR allocation query.

        Iteratively calls tools as directed by the LLM, injects results back
        into the conversation, and returns the final allocation report.

        Parameters
        ----------
        user_query : str
            Natural language request describing the disaster event.

        Returns
        -------
        str
            The agent's final allocation report as a plain-text string.
        """
        messages = self._build_initial_messages(user_query)
        _tool_cache: dict[str, str] = {}
        rag_context_added = False
        tool_round = 0

        def call_tool(name: str, args: dict) -> tuple[str, bool]:
            """
            Execute a named tool, returning its JSON result and a cache-hit flag.

            Parameters
            ----------
            name : str
                Tool function name (must be in TOOL_DISPATCH).
            args : dict
                Arguments to pass to the tool.

            Returns
            -------
            tuple[str, bool]
                (json_result, from_cache) — from_cache is True when the result
                was served from the in-session deduplication cache.
            """
            cache_key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if cache_key in _tool_cache:
                return _tool_cache[cache_key], True
            result = dispatch_tool(name, args)
            _tool_cache[cache_key] = result
            return result, False

        def call_llm(model: str) -> object:
            """
            Submit the current message history to the OpenAI chat completions API.

            Parameters
            ----------
            model : str
                Model identifier to use (e.g., 'gpt-4o' or 'gpt-4o-mini').

            Returns
            -------
            object
                OpenAI ChatCompletion response object.
            """
            estimated_tokens = _estimate_message_tokens(messages)
            if self.verbose:
                print(f"[LLM] model={model} estimated_prompt_tokens={estimated_tokens}")
            if estimated_tokens > TOKEN_WARNING_THRESHOLD and self.verbose:
                print(f"[Warning] Estimated prompt exceeds {TOKEN_WARNING_THRESHOLD} tokens.")
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )

        while tool_round < MAX_TOOL_ROUNDS:
            try:
                response = call_llm(PRIMARY_MODEL)
            except RateLimitError:
                if self.verbose:
                    print(f"[LLM] {PRIMARY_MODEL} rate limited; retrying with {FALLBACK_MODEL}")
                response = call_llm(FALLBACK_MODEL)

            if self.verbose and response.usage:
                print(
                    "[LLM] actual_tokens="
                    f"prompt:{response.usage.prompt_tokens} "
                    f"completion:{response.usage.completion_tokens} "
                    f"total:{response.usage.total_tokens}"
                )

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # Append the assistant's response (may contain tool calls or text)
            messages.append(choice.message.model_dump())

            if finish_reason == "stop":
                # No more tool calls — return the final text response
                return choice.message.content or ""

            if finish_reason == "tool_calls":
                # Execute each requested tool call
                tool_calls = choice.message.tool_calls or []
                data_tools_seen = False
                for tc in tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    result_json, from_cache = call_tool(fn_name, fn_args)
                    raw_result = json.loads(result_json)
                    compact_result = compact_tool_result(fn_name, raw_result)
                    compact_json = json.dumps(compact_result, default=str)

                    if self.verbose and not from_cache:
                        print(f"[Tool] {fn_name}({fn_args})")
                        preview = compact_json[:200] + "..." if len(compact_json) > 200 else compact_json
                        print(f"[Result] {preview}\n")

                    if fn_name in {"nri_lookup", "svi_lookup", "census_lookup"}:
                        data_tools_seen = True

                    # Inject tool result into message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": compact_json,
                    })

                    # Respect rate limits with a short pause between tool calls
                    if not from_cache:
                        time.sleep(0.2)

                if data_tools_seen and not rag_context_added:
                    context_block = retrieve_context(
                        "Interpret FEMA NRI risk, CDC SVI percentile, poverty rate, and housing burden for disaster recovery targeting.",
                        k=2,
                        corpus_dir=self.corpus_dir,
                    )
                    if context_block:
                        messages.append({
                            "role": "user",
                            "content": (
                                "Use this retrieved methodology context when interpreting the collected "
                                "NRI, SVI, poverty, and housing burden values in the final report. "
                                "Do not call additional tools just to retrieve methodology context.\n\n"
                                f"{context_block}"
                            ),
                        })
                    rag_context_added = True

                tool_round += 1
            else:
                # Unexpected finish reason — return whatever text is available
                return choice.message.content or f"Agent stopped with reason: {finish_reason}"

        return (
            "Maximum tool call rounds reached. The agent was unable to complete the full "
            "analysis. Please try with a smaller disaster event or fewer parishes."
        )

    def get_token_usage(self, user_query: str) -> dict:
        """
        Estimate token usage for a single run without executing tools.
        Useful for cost projection before running a full analysis.

        Parameters
        ----------
        user_query : str
            The user's query.

        Returns
        -------
        dict
            Estimated prompt token count and projected cost at GPT-4o rates.
        """
        messages = self._build_initial_messages(user_query)
        # Rough estimate: 4 chars per token
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = total_chars // 4
        cost_estimate_usd = (estimated_tokens / 1_000_000) * 5.0  # GPT-4o input rate

        return {
            "estimated_prompt_tokens": estimated_tokens,
            "estimated_cost_usd": round(cost_estimate_usd, 4),
        }
