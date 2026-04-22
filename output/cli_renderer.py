"""
Phase 5 — CLI Renderer

Render recommendations as beautiful terminal cards using Rich.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Rank medals
_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def render_cards(formatted_recs: list[dict]) -> None:
    """Render recommendation cards to the terminal."""
    if not formatted_recs:
        console.print("\n[bold red]😔 No recommendations found.[/bold red]")
        console.print("Try broadening your search criteria.\n")
        return

    console.print()
    console.rule("[bold cyan]🍽️  Your Personalized Recommendations[/bold cyan]")
    console.print()

    for rec in formatted_recs:
        medal = _MEDALS.get(rec["rank"], f"#{rec['rank']}")
        title = f"{medal}  {rec['name']}"

        # Build card content
        lines = [
            f"🍕 [bold]Cuisines[/bold]    {rec['cuisines']}",
            f"⭐ [bold]Rating[/bold]      {rec['rating']}",
            f"💰 [bold]Cost[/bold]        {rec['cost']} for two",
            f"📍 [bold]Location[/bold]    {rec['location']}",
            "",
            f"🤖 [bold]Why this?[/bold]",
            f"   {rec['explanation']}",
        ]

        if rec.get("trade_offs"):
            lines.append(f"")
            lines.append(f"⚠️  [dim]{rec['trade_offs']}[/dim]")

        content = "\n".join(lines)

        console.print(Panel(
            content,
            title=f"[bold white]{title}[/bold white]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

    console.rule("[dim]End of recommendations[/dim]")
    console.print()
