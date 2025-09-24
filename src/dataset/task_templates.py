EASY_TASKS = [
    {"user_prompt": "Get the current US market status from Polygon.", "available_tools": ["polygon"], "max_turns": 4, "success_criteria": {"must_call_tool": "polygon", "max_tool_calls": 2}}
]
MEDIUM_TASKS = [
    {"user_prompt": "Fetch AAPL fundamentals from FMP and summarize key ratios.", "available_tools": ["fmp"], "max_turns": 6, "success_criteria": {"must_call_tool": "fmp", "max_tool_calls": 3}}
]
HARD_TASKS = [
    {"user_prompt": "Search Tavily for Nvidia earnings preview and compile a 3-bullet summary.", "available_tools": ["tavily"], "max_turns": 8, "success_criteria": {"must_call_tool": "tavily", "max_tool_calls": 3}}
]
