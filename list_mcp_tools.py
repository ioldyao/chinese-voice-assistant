import asyncio
from src.voice_assistant.mcp_client import MCPManager

async def test():
    mcp = MCPManager()
    await mcp.add_server_async('playwright', 'npx', ['@playwright/mcp@latest'], 120)
    tools = await mcp.list_all_tools_async()

    playwright_tools = [t for t in tools if t.get('server') == 'playwright']
    for tool in playwright_tools:
        print(f"{tool['name']}: {tool['description']}")

    await mcp.stop_all_async()

asyncio.run(test())
