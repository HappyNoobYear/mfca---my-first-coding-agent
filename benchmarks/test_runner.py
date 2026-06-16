"""Test runner for agent comparison benchmarks."""

import asyncio
import json
from typing import Dict, List
from pathlib import Path

from benchmarks.agent_interface import AgentInterface, AgentResult
from benchmarks.adapters.mfca_adapter import MFCAAdapter

try:
    from benchmarks.adapters.smolagents_adapter import SmolagentsAdapter
except ImportError as e:
    print(f"Warning: smolagents not available: {e}")
    SmolagentsAdapter = None

try:
    from benchmarks.adapters.ohpi_adapter import OhMyPiAdapter
except ImportError as e:
    print(f"Warning: oh-my-pi not available: {e}")
    OhMyPiAdapter = None


class BenchmarkRunner:
    """Orchestrates benchmarking across all agents."""

    def __init__(self):
        self.adapters = {}
        self.results = {}

    async def initialize_agents(self, agents: List[str] = None):
        """Initialize selected agents."""
        if agents is None:
            agents = ["mfca", "smolagents", "ohpi"]

        if "mfca" in agents:
            try:
                self.adapters["mfca"] = MFCAAdapter()
                await self.adapters["mfca"].initialize()
                print("✓ Initialized mfca")
            except Exception as e:
                print(f"✗ Failed to initialize mfca: {e}")

        if "smolagents" in agents and SmolagentsAdapter:
            try:
                self.adapters["smolagents"] = SmolagentsAdapter()
                await self.adapters["smolagents"].initialize()
                print("✓ Initialized smolagents")
            except Exception as e:
                print(f"✗ Failed to initialize smolagents: {e}")
        elif "smolagents" in agents:
            print("⊘ smolagents not available (not installed)")

        if "ohpi" in agents and OhMyPiAdapter:
            try:
                self.adapters["ohpi"] = OhMyPiAdapter()
                await self.adapters["ohpi"].initialize()
                print("✓ Initialized oh-my-pi")
            except Exception as e:
                print(f"✗ Failed to initialize oh-my-pi: {e}")
        elif "ohpi" in agents:
            print("⊘ oh-my-pi not available")

    async def run_scenario(self, name: str, description: str, turns: List[str] = None):
        """Run a scenario across all agents."""
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"Description: {description}")
        print(f"{'='*60}")

        results = {}

        for agent_name, adapter in self.adapters.items():
            try:
                print(f"\nTesting {agent_name}...", end=" ", flush=True)

                if turns and len(turns) > 1:
                    result = await adapter.process_multi_turn(turns)
                else:
                    task = turns[0] if turns else description
                    result = await adapter.process_task(task)

                results[agent_name] = result
                print(f"✓ ({result.tokens_used} tokens, {result.tool_calls} tools)")

                if not result.success:
                    print(f"  Error: {result.error}")

            except Exception as e:
                print(f"✗ Error: {e}")
                results[agent_name] = AgentResult(
                    success=False,
                    output="",
                    tokens_used=0,
                    tool_calls=0,
                    turns_completed=0,
                    error=str(e),
                )

        self.results[name] = results
        self._print_scenario_summary(name, results)

    def _print_scenario_summary(self, scenario: str, results: Dict[str, AgentResult]):
        """Print summary of scenario results."""
        print(f"\n{'─'*60}")
        print("Summary:")
        print(f"{'Agent':<15} {'Tokens':<10} {'Tools':<8} {'Success':<10}")
        print(f"{'─'*60}")

        for agent_name, result in results.items():
            status = "✓" if result.success else "✗"
            print(
                f"{agent_name:<15} {result.tokens_used:<10} "
                f"{result.tool_calls:<8} {status:<10}"
            )

    async def run_all_scenarios(self):
        """Run all benchmark scenarios."""
        scenarios = [
            (
                "simple_file_read",
                "Read a file and explain it",
                ["Read the file /app/AGENT_CONTEXT.md and summarize what it contains"],
            ),
            (
                "multi_turn_5",
                "5-turn code modification scenario",
                [
                    "Create a simple Python script that prints numbers from 1 to 5",
                    "Modify it to print only even numbers",
                    "Add a variable to count iterations",
                    "Make it more concise",
                    "Show me the final code",
                ],
            ),
        ]

        for scenario_name, description, turns in scenarios:
            await self.run_scenario(scenario_name, description, turns)

    async def cleanup_agents(self):
        """Clean up all agents."""
        for adapter in self.adapters.values():
            try:
                await adapter.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")

    def export_results(self, output_file: str = "results/benchmark_results.json"):
        """Export results to JSON."""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        export_data = {}
        for scenario, results in self.results.items():
            export_data[scenario] = {
                agent: {
                    "success": result.success,
                    "tokens_used": result.tokens_used,
                    "tool_calls": result.tool_calls,
                    "turns_completed": result.turns_completed,
                    "error": result.error,
                }
                for agent, result in results.items()
            }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Results exported to {output_file}")

    def print_final_report(self):
        """Print final comparison report."""
        print("\n" + "=" * 60)
        print("FINAL BENCHMARK REPORT")
        print("=" * 60)

        if not self.results:
            print("No results to report")
            return

        # Calculate averages
        agent_stats = {}
        for scenario, results in self.results.items():
            for agent_name, result in results.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        "total_tokens": 0,
                        "total_tools": 0,
                        "scenarios": 0,
                        "successes": 0,
                    }

                agent_stats[agent_name]["scenarios"] += 1
                agent_stats[agent_name]["total_tokens"] += result.tokens_used
                agent_stats[agent_name]["total_tools"] += result.tool_calls
                if result.success:
                    agent_stats[agent_name]["successes"] += 1

        print(f"\n{'Agent':<15} {'Avg Tokens':<15} {'Avg Tools':<12} {'Success Rate':<12}")
        print("─" * 60)

        for agent_name, stats in sorted(agent_stats.items()):
            avg_tokens = stats["total_tokens"] / max(stats["scenarios"], 1)
            avg_tools = stats["total_tools"] / max(stats["scenarios"], 1)
            success_rate = (
                stats["successes"] / max(stats["scenarios"], 1) * 100
            )

            print(
                f"{agent_name:<15} {avg_tokens:<15.0f} "
                f"{avg_tools:<12.1f} {success_rate:<12.0f}%"
            )


async def main():
    """Main entry point."""
    runner = BenchmarkRunner()

    try:
        # Initialize agents
        await runner.initialize_agents()

        # Run all scenarios
        await runner.run_all_scenarios()

        # Export results
        runner.export_results()

        # Print final report
        runner.print_final_report()

    finally:
        await runner.cleanup_agents()


if __name__ == "__main__":
    asyncio.run(main())
