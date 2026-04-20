import asyncio
from orchestrator.auto_trainer import AutoTrainerManager

class MockDocker:
    def list_active_bots(self):
        return [{"symbol": "AAPL"}]
    def list_completed_bots(self):
        return [{"symbol": "AAPL"}]

async def main():
    mgr = AutoTrainerManager(MockDocker())
    mgr.start()
    await asyncio.sleep(1)

asyncio.run(main())
