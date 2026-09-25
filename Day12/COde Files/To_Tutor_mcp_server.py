
from fastmcp import FastMCP


# # ── CREATE THE SERVER ────────────────────────────────────────

mcp = FastMCP("CompanyKB")


# # ── SAMPLE DATA ──────────────────────────────────────────────

EMPLOYEES = {
    "engineering": ["Priya Sharma", "Rahul Verma", "Anita Desai"],
    "product": ["Vikram Iyer", "Sneha Nair"],
    "data_science": ["Amit Kulkarni", "Deepa Joshi", "Karan Mehta"],
}

TEAM_PROFILES = {
    "priya": {
        "name": "Priya Sharma", 
        "role": "Senior Backend Engineer",
        "department": "Engineering",
        "projects": ["Payment Gateway v2", "MCP Integration"],
    },
    "amit": {
        "name": "Amit Kulkarni",
        "role": "Lead Data Scientist",
        "department": "Data Science",
        "projects": ["Fraud Detection Model", "Customer Churn Prediction"],
    },
}


# # ============================================================
# # PRIMITIVE 1: RESOURCES (data the AI can READ)
# # ============================================================

@mcp.resource("company://info")
def get_company_info() -> str:
    """General information about the company."""
    return (
        "Company: TechNova Solutions Pvt. Ltd.\n"
        "Founded: 2018, Pune, India\n"
        "Departments: Engineering, Product, Data Science\n"
        "Team Size: 45 employees\n"
        "Focus: AI-powered fintech products for the Indian market"
    )


# # ── Resource 2: Team Profiles (dynamic) ─────────────────────

@mcp.resource("company://team/{name}")
def get_team_profile(name: str) -> str:
    """Profile information for a specific team member."""
    # name.lower() → normalize to match our dictionary keys
    profile = TEAM_PROFILES.get(name.lower())

    if profile is None:
        return f"No profile found for '{name}'"

    return (
        f"Name: {profile['name']}\n"
        f"Role: {profile['role']}\n"
        f"Department: {profile['department']}\n"
        f"Current Projects: {', '.join(profile['projects'])}"
    )


# # ============================================================
# # PRIMITIVE 2: TOOLS (actions the AI can CALL)
# # ============================================================

@mcp.tool
def search_employees(department: str) -> str:
    """Search for employees in a specific department.
    Valid departments: engineering, product, data_science."""

    # department.lower() → normalize user input
    dept = department.lower()
    employees = EMPLOYEES.get(dept)

    if employees is None:
        valid = ", ".join(EMPLOYEES.keys())
        return f"Department '{department}' not found. Valid: {valid}"

    result = f"Employees in {department}:\n"
    for i in range(len(employees)):
        result = result + f"  {i + 1}. {employees[i]}\n"

    return result


# # ── Tool 2: Estimate Project Cost ───────────────────────────
# #

@mcp.tool
def estimate_project_cost(team_size: int, duration_weeks: int) -> str:
    """Estimate the cost of a project based on team size and duration.
    Uses average Indian tech salary rates."""

    # Rs 3,00,000/month per engineer (average, Indian market)
    monthly_cost_per_person = 300000
    weekly_cost_per_person = monthly_cost_per_person / 4

    total_cost = team_size * weekly_cost_per_person * duration_weeks

    return (
        f"Project Cost Estimate:\n"
        f"  Team Size: {team_size} engineers\n"
        f"  Duration: {duration_weeks} weeks\n"
        f"  Weekly cost per person: Rs {weekly_cost_per_person:,.0f}\n"
        f"  Total Estimated Cost: Rs {total_cost:,.0f}\n"
        f"  (Based on avg Indian tech salary of Rs 3,00,000/month)"
    )


# # ============================================================
# # PRIMITIVE 3: PROMPTS (templates the USER triggers)
# # ============================================================

@mcp.prompt()
def onboarding_brief(employee_name: str, department: str) -> str:
    """Create a structured onboarding plan for a new employee."""
    return (
        f"Create a 30-day onboarding plan for {employee_name} "
        f"who is joining the {department} department.\n\n"
        f"Include:\n"
        f"1. Week 1: Setup and introductions\n"
        f"2. Week 2: Core tool training\n"
        f"3. Week 3: First small project\n"
        f"4. Week 4: Review and feedback\n\n"
        f"Use the company info and team profiles available "
        f"as resources to personalize the plan."
    )


# # ── START THE SERVER ─────────────────────────────────────────

if __name__ == "__main__":
    print("Starting CompanyKB MCP Server...")
    print("Resources: company://info, company://team/{name}")
    print("Tools: search_employees, estimate_project_cost")
    print("Prompts: onboarding_brief")
    mcp.run()

