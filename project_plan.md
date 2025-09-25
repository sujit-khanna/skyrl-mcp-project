# Order of execution for this project

## Setup Ex SkyRL

    1. Initial setup: excluding skyrl (only for synthetic data generation pipelines)
       DONE 

    2. Create and test MCP servers

    3. Create synthetic data generation pipeline (including validation)

## Setup with SkyRL dependencies

    1. Create and test the multi-mcp tool multi-turn SkyRL environment
    2. Creat reward functions:
        a. Heuristic based like with DeepSeek-R1
        b. LLM-as-a-Judge form of reward model
    3. Creating training scripts via GRPO/PPO
    4. Traning monitoring and logging