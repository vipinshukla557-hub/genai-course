
# # ── IMPORTS ──────────────────────────────────────────────────

import asyncio
from fastmcp import Client

# # ── THE MAIN FUNCTION ────────────────────────────────────────


async def main():

    client = Client("To_Tutor_mcp_server.py")

    async with client:

#         # ── DISCOVER TOOLS ───────────────────────────────────

        print("=" * 50)
        print("DISCOVERED TOOLS:")
        print("=" * 50)

        tools = await client.list_tools()
        for tool in tools:
            print(f"  Name: {tool.name}")
            print(f"  Description: {tool.description}")
            print()

#         # ── DISCOVER RESOURCES ───────────────────────────────

        print("=" * 50)
        print("DISCOVERED RESOURCES:")
        print("=" * 50)
        resources = await client.list_resources()
        for resource in resources:
            print(f"  URI: {resource.uri}")
            print(f"  Description: {resource.description}")
            print()

#         # ── CALL A TOOL ──────────────────────────────────────

        print("=" * 50)
        print("CALLING TOOL: search_employees")
        print("=" * 50)
        result = await client.call_tool(
            "search_employees",
            {"department": "engineering"}
        )
        print(result)

#         # ── READ A RESOURCE ──────────────────────────────────

        print("=" * 50)
        print("READING RESOURCE: company://team/priya")
        print("=" * 50)
        
        resource_data = await client.read_resource("company://team/priya")
        print(resource_data)

#         # ── DISCOVER PROMPTS ─────────────────────────────────

        print("=" * 50)
        print("DISCOVERED PROMPTS:")
        print("=" * 50)
        prompts = await client.list_prompts()
        for prompt in prompts:
            print(f"  Name: {prompt.name}")
            print(f"  Description: {prompt.description}")
            print()


# # ── RUN THE CLIENT ───────────────────────────────────────────

asyncio.run(main())

