from langchain.tools import tool
import os, json

# --------------------------------------------------------------------
# JSON PATH UTILITIES
# --------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


JS = lambda *p: os.path.join(BASE_DIR, "data", "json_store", *p)


def _load(path):
    """Utility loader function.
    Loads JSON data from the given absolute file path if it exists,
    else returns an empty list.

    Args:
        path (str): Absolute path to the JSON file.

    Returns:
        list | dict: Parsed JSON content, or an empty list if not found.
    """
    return json.load(open(path, "r", encoding="utf8")) if os.path.exists(path) else []


# --------------------------------------------------------------------
# IMPACT AGENT TOOLS
# --------------------------------------------------------------------

@tool
def get_dependencies_for_shipment(shipment_id: str) -> str:
    """
    Retrieve all downstream dependencies for a specific shipment.

    This function looks up relationships that indicate how a delay or
    disruption in a given shipment may impact dependent downstream orders,
    manufacturing units, or retailers.

    Data Source:
        downstream_dependencies.json

    Args:
        shipment_id (str): Unique ID of the shipment (e.g., "SHP1001").

    Returns:
        str: JSON-formatted string representing the dependent nodes,
             containing shipment_id, dependent_order_ids, and impact_level.
             Returns "NOT_FOUND" if the shipment ID does not exist.
    """
    data = _load(JS("downstream_dependencies.json"))
    shipment_id = str(shipment_id).strip().replace('"', '')
    for row in data:
        print("Row==>> shipment data ==>> ",row)
        if str(row.get("shipment_id")) == str(shipment_id):
            return json.dumps(row, indent=2)
    return "NOT_FOUND"


@tool
def get_open_orders(sku_region: str) -> str:
    """
    Retrieve all open customer orders for a given SKU and region.

    This tool helps the agent assess the demand pressure on inventory or
    fulfillment capacity when a delay or disruption occurs.

    Input Format:
        "SKU|Region"

    Example:
        "PROD_A100|South"

    Data Source:
        open_orders.json

    Args:
        sku_region (str): Combined SKU and region string separated by '|'.

    Returns:
        str: JSON-formatted list of open orders, each including
             order_id, sku, region, quantity, and expected_delivery_date.
             Returns "NOT_FOUND" if no matching orders are found.
    """
    sku_region = str(sku_region).strip().replace('"', '')
    try:
        sku, region = [x.strip() for x in str(sku_region).split("|", 1)]
    except:
        return "ERROR: use 'SKU|Region'"
    data = _load(JS("open_orders.json"))
    res = [o for o in data if o.get("sku") == sku and o.get("region") == region]
    return json.dumps(res, indent=2) if res else "NOT_FOUND"


@tool
def get_inventory_position(sku_loc: str) -> str:
    """
    Retrieve the real-time inventory position for a specific SKU at a given location.

    Useful for determining whether alternative inventory buffers exist to absorb
    a supply disruption.

    Input Format:
        "SKU|Location"

    Example:
        "PROD_B200|WH_Bangalore"

    Data Source:
        inventory_positions.json

    Args:
        sku_loc (str): Combined SKU and location string separated by '|'.

    Returns:
        str: JSON object with inventory fields like:
             {"sku": "PROD_B200", "location": "WH_Bangalore",
              "available_qty": 1200, "safety_stock": 200, "reserved": 150}
             Returns "NOT_FOUND" if no match is found.
    """
    sku_loc = str(sku_loc).strip().replace('"', '')
    try:
        sku, loc = [x.strip() for x in str(sku_loc).split("|", 1)]
    except:
        return "ERROR: use 'SKU|Location'"
    data = _load(JS("inventory_positions.json"))
    for r in data:
        if r.get("sku") == sku and r.get("location") == loc:
            return json.dumps(r, indent=2)
    return "NOT_FOUND"


@tool
def get_substitutions(sku: str) -> str:
    """
    Retrieve substitution options for a product SKU.

    This helps the impact assessment agent evaluate possible alternative SKUs
    that can be used to mitigate stockouts or delays.

    Data Source:
        substitutions.json

    Args:
        sku (str): Product SKU code (e.g., "PROD_C300").

    Returns:
        str: JSON-formatted object containing primary SKU and its substitute SKUs.
             Example:
             {
               "sku": "PROD_C300",
               "substitutes": ["PROD_C301", "PROD_C310"],
               "substitution_cost_factor": 1.05
             }
             Returns "NOT_FOUND" if no substitutes exist.
    """
    sku = str(sku).strip().replace('"', '')
    data = _load(JS("substitutions.json"))
    for r in data:
        if r.get("sku") == sku:
            return json.dumps(r, indent=2)
    return "NOT_FOUND"


@tool
def get_price_cost(sku: str) -> str:
    """
    Retrieve pricing and cost structure for a given SKU.

    The data is used by the impact assessment agent to estimate financial loss,
    lost sales, or cost escalation when disruptions occur.

    Data Source:
        price_cost.json

    Args:
        sku (str): Product SKU code (e.g., "PROD_A100").

    Returns:
        str: JSON object containing keys like:
             {"sku": "PROD_A100", "unit_price": 450, "unit_cost": 300, "margin_pct": 33.3}
             Returns "NOT_FOUND" if SKU data is unavailable.
    """
    sku = str(sku).strip().replace('"', '')
    data = _load(JS("price_cost.json"))
    for r in data:
        if r.get("sku") == sku:
            return json.dumps(r, indent=2)
    return "NOT_FOUND"


@tool
def get_transport_rate(route: str) -> str:
    """
    Retrieve transportation rate information for a given origin-destination route.

    Used to calculate expedited freight costs or rerouting options during impact analysis.

    Input Format:
        "Origin->Destination"

    Example:
        "Factory_01->Retail_Delhi"

    Data Source:
        transport_rates.json

    Args:
        route (str): Route string with origin and destination separated by '->'.

    Returns:
        str: JSON object containing:
             {"route": "Factory_01->Retail_Delhi", "carrier": "Safexpress",
              "base_rate_per_kg": 12.5, "expedite_multiplier": 1.4}
             Returns "NOT_FOUND" if route data is missing.
    """
    route = str(route).strip().replace('"', '')
    data = _load(JS("transport_rates.json"))
    for r in data:
        if r.get("route") == route:
            return json.dumps(r, indent=2)
    return "NOT_FOUND"


@tool
def get_sla_policy(customer_tier: str) -> str:
    """
    Retrieve SLA (Service Level Agreement) penalties and cost policies for a customer tier.

    SLA tiers define penalties, stockout costs, and cancellation percentages.
    This information helps the agent quantify contractual penalties and
    financial impact when delivery SLAs are breached.

    Data Source:
        sla_penalties.json

    Args:
        customer_tier (str): Customer category (e.g., "Platinum", "Gold", "Silver").

    Returns:
        str: JSON object containing SLA parameters such as:
             {
               "tier": "Platinum",
               "sla_hours": 48,
               "penalty_rate_per_hour": 1000,
               "stockout_cost_per_unit": 150,
               "cancellation_penalty_pct": 10
             }
             Returns "NOT_FOUND" if the tier is not recognized.
    """

    customer_tier = str(customer_tier).strip().replace('"', '')
    data = json.load(open(JS("sla_penalties.json"), "r", encoding="utf8")) if os.path.exists(JS("sla_penalties.json")) else {}
    tiers = data.get("tiers", {})
    if customer_tier in tiers:
        return json.dumps({
            "tier": customer_tier,
            **tiers[customer_tier],
            "stockout_cost_per_unit": data.get("stockout_cost_per_unit", 0),
            "cancellation_penalty_pct": data.get("cancellation_penalty_pct", 0)
        }, indent=2)
    return "NOT_FOUND"
