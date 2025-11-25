
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description
from langchain_core.agents import AgentFinish, AgentAction
from langchain_openai import ChatOpenAI

from services.llm_factory import create_chat_model
from services.logger_config import get_logger
from util.guardrail_util import load_guardrails
import json as js

logger = get_logger(__name__)

from agents.tools.impact_agent_tools import get_dependencies_for_shipment, get_open_orders, get_inventory_position, \
    get_substitutions, get_price_cost, get_transport_rate, get_sla_policy
from util.ragas_util import record_agent_run_for_ragas

IMPACT_TOOLS = [
    get_dependencies_for_shipment, get_open_orders, get_inventory_position,
    get_substitutions, get_price_cost, get_transport_rate, get_sla_policy
]


def build_impact_agent(tools: list[Tool], model="gpt-4o-mini", temperature=0.2):
    guard = load_guardrails()  # 👈 Inject latest configuration

    guardrail_prompt = f"""
    ⚙️ **Supply Chain Guardrails & Compliance:**
    - Minimum confidence for recommendations: {guard.get('min_confidence_for_auto_action', 70)}%
    - Banned phrases: {', '.join(guard.get('banned_phrases', []))}
    - Business rules:
      {"; ".join(guard.get('business_rules', []))}

    Ensure all reasoning, impact assessments strictly comply with these guardrails.
    If an action violates a rule, explicitly mention "⚠️ REJECTED_BY_GUARDRAIL" and propose an alternative.
    """


#     template = """
#     You are a Supply Chain Impact Analysis expert.
#     You are investigating Impact of an alert about disruption.
# Given: an Alert + RCA result. Estimate downstream business impact and costs.
#
# You can use these tools:
# {tools}
#
# {guardrail_prompt}
#     When you use a tool, please follow this format:
#
#     Question: the input alert details or question
#     Thought: think about what to check next
#     Action: one of [{tool_names}]
#     Action Input: the input to the tool
#     Observation: result of the tool
#     ... (repeat Thought/Action/Action Input/Observation as required)
#     Thought: I now know the root cause
#     Final Answer: A JSON object only in the format below ::
#     JSON structure:
#     {{
#       "impact_summary": "One paragraph summary of business impact.",
#       "delay_hours": float,
#       "items": [
#         {{
#           "sku": "string",
#           "region": "string",
#           "at_risk_units": int,
#           "unfulfilled_units": int,
#           "lost_sales": float,
#           "sla_penalty": float,
#           "expedite_cost": float,
#           "notes": "string"
#         }}
#       ],
#       "total": {{
#         "lost_sales": float,
#         "sla_penalty": float,
#         "expedite_cost": float,
#         "overall": float
#       }},
#       "assumptions": [
#         "Explicit list of assumptions used to estimate impact",
#         "Include substitution, buffer, and SLA policy considerations"
#       ],
#       "confidence": float
#     }}
#
#     Rules:
#     - You must always produce a valid JSON object matching the above schema.
#     - Do not include Markdown, text commentary, or any extra output.
#     - Ground every conclusion in tool outputs or given context.
#     - If insufficient data, set numeric values to 0 and explain assumptions.
#
#
# Do not hallucinate, if you dont get correct appropriate context.
# Be explicit in assumptions. Use substitutions, buffers, SLA tiers where possible.
# Be thorough and ground reasoning to the tool observations.
#
#     Begin!
#
#     Question: {input}
#     Thought: {agent_scratchpad}
# """

    template = """
       You are a Supply Chain Impact Analysis expert.
Your task is to calculate the quantitative business impact of a given alert and RCA result.

You are provided with:
- An alert describing a disruption event.
- The RCA (Root Cause Analysis) result detailing cause and recommendations.

Follow these strict steps and rules while reasoning:

1️⃣ **Information Gathering**
   - Use tools to fetch dependencies, open orders, inventory, substitutions, price, transport rates, and SLA policies.
   - Combine findings logically; avoid redundant tool calls.

2️⃣ **Calculation Rules**
   Use these formulas consistently for all items:
   - **At Risk Units** = dependent or pending downstream orders tied to this shipment.
   - **Unfulfilled Units** = min(At Risk Units, Available Inventory Shortage).
   - **Lost Sales (Rs)** = Unfulfilled Units × Unit Selling Price (from price_cost.json).
   - **SLA Penalty (Rs)** = (Unfulfilled Units × SLA penalty per unit from SLA policy).
   - **Expedite Cost (Rs)** = If delay_hours > SLA threshold, Expedite Cost = At Risk Units × transport rate × 0.10 (10% premium).
   - **Overall Impact (Rs)** = Lost Sales + SLA Penalty + Expedite Cost.

3️⃣ **Delay Estimation**
   - Base delay_hours on available shipment or RCA details.
   - If not available, use default 12 hours for medium disruptions and 24 for severe disruptions.

4️⃣ **Assumptions & Confidence**
   - Always list assumptions (pricing basis, SLA tier, inventory substitution, delay logic).
   - Confidence:
       - 0.9 if data sourced from ≥3 tools.
       - 0.7 if only partial data available.
       - 0.5 if many assumptions.
       

    You can use these tools:
    {tools}

    {guardrail_prompt}
        When you use a tool, please follow this format:

        Question: the input alert details or question
        Thought: think about what to check next
        Action: one of [{tool_names}]
        Action Input: the input to the tool
        Observation: result of the tool
        ... (repeat Thought/Action/Action Input/Observation as required)
        Thought: I now know the root cause
        Final Answer: A JSON object only in the format below ::
        JSON structure:
        {{
          "impact_summary": "One paragraph summary of business impact.",
          "delay_hours": float,
          "items": [
            {{
              "sku": "string",
              "region": "string",
              "at_risk_units": int,
              "unfulfilled_units": int,
              "lost_sales": float,
              "sla_penalty": float,
              "expedite_cost": float,
              "notes": "string"
            }}
          ],
          "total": {{
            "lost_sales": float,
            "sla_penalty": float,
            "expedite_cost": float,
            "overall": float
          }},
          "assumptions": [
            "Explicit list of assumptions used to estimate impact",
            "Include substitution, buffer, and SLA policy considerations"
          ],
          "confidence": float
        }}


       
        Rules:
        - You must always produce FINAL ANSWER in  a valid JSON object matching the above schema.
        - Do not include Markdown, text commentary, or any extra output.
        - Ground every conclusion in tool outputs or given context.
        - If insufficient data, set numeric values to 0 and explain assumptions.


    Do not hallucinate, if you dont get correct appropriate context.
    Be explicit in assumptions. Use substitutions, buffers, SLA tiers where possible.
    Be thorough and ground reasoning to the tool observations.

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        guardrail_prompt=guardrail_prompt,
        tool_names=", ".join([t.name for t in tools]),
    )
    # llm = ChatOpenAI(model=model, temperature=temperature, stop=["\nObservation", "Observation"])
    llm = create_chat_model(is_agent=True)
    return (
            {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])}
            | prompt | llm | ReActSingleInputOutputParser()
    )


def run_impact_agent(alert: dict, rca_json: dict, max_steps=8, llm_model="gpt-4o-mini", temperature=0.2, tools=None):
    """Run the Impact Assessment agent with trace and RAGAS logging."""

    run_trace, agent_scratchpad, retrieved_contexts = [], [], []
    final_text, final_json, raw_agent_return = "", None, None
    tools = tools or IMPACT_TOOLS

    try:
        agent = build_impact_agent(tools, model=llm_model, temperature=temperature)
        question = f"Given this alert and RCA result, estimate downstream business and cost impact.\nAlert: {js.dumps(alert)}\nRCA: {js.dumps(rca_json)}"

        steps = 0
        while steps < max_steps:
            agent_step = agent.invoke({"input": question, "agent_scratchpad": agent_scratchpad})
            if isinstance(agent_step, AgentAction):
                tool_name, tool_input = agent_step.tool, agent_step.tool_input
                try:
                    tool_obj = next(t for t in tools if t.name == tool_name)
                    observation = tool_obj.func(str(tool_input))
                except Exception as e:
                    observation = f"ERROR: {e}"

                run_trace.append({
                    "step": steps + 1,
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "observation": observation
                })
                agent_scratchpad.append((agent_step, observation))
                retrieved_contexts.append(str(observation))
                steps += 1
                continue

            if isinstance(agent_step, AgentFinish):
                final_text = agent_step.return_values.get("output", "")
                print("Final text ==>>> ", final_text)

                # --- Safe JSON parsing with cleaning ---
                import re, json

                try:
                    # Extract potential JSON region
                    match = re.search(r'(\{.*\}|\[.*\])', final_text, re.DOTALL)
                    json_str = match.group(0) if match else final_text

                    # Basic cleanup to handle truncations or bad quotes
                    json_str = json_str.strip()
                    json_str = json_str.replace("\n", " ").replace("\\n", " ")
                    json_str = json_str.replace("“", '"').replace("”", '"')

                    # Try parsing
                    final_json = json.loads(json_str)
                except Exception as e:
                    logger.warning(f"[IMPACT_AGENT] JSON parsing failed: {e} | raw={final_text[:500]}")
                    final_json = {"_parse_error": str(e), "raw_output": final_text}

                # ✅ Log to RAGAS dataset for quality evaluation
                record_agent_run_for_ragas(
                    agent_name="Impact_Agent",
                    alert=alert,
                    question=question,
                    contexts=retrieved_contexts,
                    answer=final_text
                )

                return {
                    "steps": run_trace,
                    "final_answer": final_text,
                    "final_json": final_json,
                    "logs": format_log_to_str(agent_scratchpad)
                }

        return {
            "steps": run_trace,
            "final_answer": final_text,
            "final_json": final_json,
            "logs": format_log_to_str(agent_scratchpad),
            "note": "max_steps_reached"
        }

    except Exception as e:
        logger.error(f"[IMPACT_AGENT] Error: {e}")
        return {"error": str(e), "steps": run_trace, "logs": format_log_to_str(agent_scratchpad)}

