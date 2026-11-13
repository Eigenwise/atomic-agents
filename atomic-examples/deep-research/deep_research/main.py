"""
Deep-research orchestrator.

Reads like a recipe. First turn: plan → (per sub-topic) search → scrape →
extract → reflect → (maybe loop) → write. Follow-up turns in chat mode:
decider routes to either another research pass (plan → research → qa)
or straight to qa against the accumulated state.

Each step is a call to a single-purpose agent (see ``deep_research/agents/``)
that reads from and contributes to the shared ``ResearchState``.

Run:
  ``python -m deep_research "your question here"``    # one-shot report
  ``python -m deep_research``                         # interactive chat
"""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from atomic_agents import AtomicAgent, BaseIOSchema

from deep_research.agents.decider_agent import DeciderInput, decider_agent
from deep_research.agents.extractor_agent import ExtractorInput, extractor_agent
from deep_research.agents.planner_agent import PlannerInput, planner_agent
from deep_research.agents.qa_agent import QAInput, qa_agent
from deep_research.agents.reflector_agent import ReflectorInput, reflector_agent
from deep_research.agents.writer_agent import WriterInput, writer_agent
from deep_research.config import ChatConfig, ResearchBudget
from deep_research.context_providers import CurrentDateProvider, ResearchStateProvider
from deep_research.state import Learning, ResearchState, SubTopic
from deep_research.tools.searxng_search import (
    SearXNGSearchTool,
    SearXNGSearchToolConfig,
    SearXNGSearchToolInputSchema,
)
from deep_research.tools.webpage_scraper import (
    WebpageScraperTool,
    WebpageScraperToolInputSchema,
)

# Rich renders unicode liberally (→, bullets, box-drawing). On Windows the
# default stdout/stderr encoding is cp1252, so piping or redirecting output
# crashes on any non-cp1252 character. Reconfigure to utf-8 with a safe
# fallback so the example runs anywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()

# How many new sub-topics a follow-up research pass may add. Kept small so
# chat follow-ups don't balloon into full extra reports.
FOLLOW_UP_SUB_TOPICS = 2


def _run_agent[
    InputSchema: BaseIOSchema, OutputSchema: BaseIOSchema
](agent: AtomicAgent[InputSchema, OutputSchema], params: InputSchema, state: ResearchState) -> OutputSchema:
    """Reserve one call before starting it, including attempts that later fail."""
    if state.agent_calls >= ResearchBudget.hard_call_cap:
        raise RuntimeError(f"Hard call cap reached ({ResearchBudget.hard_call_cap} agent call attempts).")
    state.agent_calls += 1
    return agent.run(params)


def wire_context_providers(state: ResearchState) -> None:
    """Register the state + current-date providers on every agent.

    All agents — including the planner and the chat-mode pair (decider, qa) —
    see the live ``ResearchState``. The planner's state awareness is what
    lets follow-up re-plans extend coverage instead of duplicating it.
    """
    state_provider = ResearchStateProvider("Research State", state)
    date_provider = CurrentDateProvider("Current Date")

    for agent in (planner_agent, extractor_agent, reflector_agent, writer_agent, decider_agent, qa_agent):
        agent.register_context_provider("current_date", date_provider)
        agent.register_context_provider("research_state", state_provider)


def plan_research(state: ResearchState, num_sub_topics: int = ResearchBudget.num_sub_topics) -> list[SubTopic]:
    """Run the planner, append new sub-topics to ``state.plan``, return just the new ones."""
    before = len(state.plan)
    result = _run_agent(planner_agent, PlannerInput(question=state.question, num_sub_topics=num_sub_topics), state)

    for st in result.sub_topics:
        state.plan.append(SubTopic(name=st.name, initial_queries=list(st.initial_queries)))
        state.queries_seen.update(st.initial_queries)

    new_sub_topics = state.plan[before:]
    for i, st in enumerate(new_sub_topics, 1):
        console.print(f"  [bold]{i}. {st.name}[/bold]")
        for q in st.initial_queries:
            console.print(f"     • [dim]{q}[/dim]")
    return new_sub_topics


def search_and_scrape(
    queries: list[str],
    state: ResearchState,
    search: SearXNGSearchTool,
    scraper: WebpageScraperTool,
) -> list[tuple[str, str]]:
    """Run SearXNG on the given queries, scrape the top N new URLs, return ``[(source_id, content), …]``.

    Skips URLs we've already scraped in a previous iteration. Registers
    every new URL as a ``Source`` so downstream claims can cite by ID.
    """
    results = search.run(SearXNGSearchToolInputSchema(queries=queries, category="general"))

    scraped: list[tuple[str, str]] = []
    for r in results.results:
        if r.url in state.urls_seen:
            continue
        if len(scraped) >= ResearchBudget.scrape_top_n_per_iteration:
            break

        page = scraper.run(WebpageScraperToolInputSchema(url=r.url, include_links=False))
        if page.error or not page.content.strip():
            console.print(f"    [dim]skip {r.url}: {page.error or 'empty content'}[/dim]")
            continue

        source = state.register_source(url=r.url, title=r.title or page.metadata.title)
        scraped.append((source.id, page.content))

    return scraped


def extract_claims(sub_topic: SubTopic, scraped: list[tuple[str, str]], state: ResearchState) -> int:
    """Call the extractor once per scraped source, append claims to state, return claim count."""
    new_claim_count = 0
    for source_id, content in scraped:
        source = next(s for s in state.sources if s.id == source_id)
        result = _run_agent(
            extractor_agent,
            ExtractorInput(
                sub_topic=sub_topic.name,
                source_url=source.url,
                source_title=source.title,
                content=content[: ResearchBudget.max_extractor_content_chars],
            ),
            state,
        )

        for claim in result.claims:
            state.learnings.append(Learning(text=claim, source_id=source_id, sub_topic=sub_topic.name))
            new_claim_count += 1
    return new_claim_count


def reflect(sub_topic: SubTopic, iteration: int, state: ResearchState) -> tuple[bool, list[str]]:
    """Ask the reflector whether this sub-topic has enough material. Returns (sufficient, next_queries)."""
    result = _run_agent(
        reflector_agent,
        ReflectorInput(
            sub_topic=sub_topic.name,
            iterations_so_far=iteration,
            max_iterations=ResearchBudget.max_depth_per_sub_topic,
        ),
        state,
    )

    console.print(f"    [italic]{result.reasoning}[/italic]")

    # Dedup: reflector might suggest a query we've already tried.
    fresh = [q for q in result.next_queries if q not in state.queries_seen]
    state.queries_seen.update(fresh)
    return result.sufficient, fresh


def research_sub_topic(
    sub_topic: SubTopic,
    state: ResearchState,
    search: SearXNGSearchTool,
    scraper: WebpageScraperTool,
) -> None:
    """Run the depth loop for a single sub-topic until sufficient or out of iterations."""
    console.rule(f"[bold cyan]Sub-topic: {sub_topic.name}")

    queries = sub_topic.initial_queries
    for iteration in range(1, ResearchBudget.max_depth_per_sub_topic + 1):
        if state.agent_calls >= ResearchBudget.hard_call_cap:
            console.print("[red]Hit hard call cap — stopping this sub-topic.[/red]")
            return

        console.print(f"\n  [bold]Iteration {iteration}/{ResearchBudget.max_depth_per_sub_topic}[/bold]")
        console.print(f"    queries: {queries}")

        scraped = search_and_scrape(queries, state, search, scraper)
        console.print(f"    scraped {len(scraped)} new source(s)")

        if not scraped:
            # No new information to extract from — further iterations won't help either.
            sub_topic.sufficient = True
            return

        new_claims = extract_claims(sub_topic, scraped, state)
        console.print(f"    extracted {new_claims} claim(s)")

        sufficient, next_queries = reflect(sub_topic, iteration, state)
        if sufficient or iteration == ResearchBudget.max_depth_per_sub_topic or not next_queries:
            sub_topic.sufficient = sufficient
            return

        queries = next_queries


def write_report(state: ResearchState) -> tuple[str, str]:
    """Draft the report, then run a cheap verification pass over it. Returns (headline, report)."""
    console.rule("[bold cyan]3. Write")

    writer_agent.reset_history()
    draft = _run_agent(writer_agent, WriterInput(question=state.question, mode="draft", draft=""), state)
    console.print("  [dim]draft written, verifying citations…[/dim]")

    writer_agent.reset_history()
    verified = _run_agent(writer_agent, WriterInput(question=state.question, mode="verify", draft=draft.report), state)
    return verified.headline, verified.report


def run_initial_pipeline(question: str, state: ResearchState, search: SearXNGSearchTool, scraper: WebpageScraperTool) -> None:
    """First-turn pipeline: plan → research → write. Populates and prints state."""
    console.print(Panel.fit(f"[bold]Deep Research[/bold]\n{question}", border_style="blue"))
    state.question = question

    console.rule("[bold cyan]1. Plan")
    new_sub_topics = plan_research(state)

    console.rule("[bold cyan]2. Research")
    for sub_topic in new_sub_topics:
        research_sub_topic(sub_topic, state, search, scraper)

    headline, report = write_report(state)

    console.rule("[bold green]Report")
    console.print(Panel(f"[bold]{headline}[/bold]", border_style="green"))
    console.print(Markdown(report))
    _print_stats(state)


def run(question: str) -> None:
    """One-shot entrypoint: plan, research, write, print the report. No chat loop."""
    state = ResearchState(question=question)
    wire_context_providers(state)

    search, scraper = _build_tools()
    run_initial_pipeline(question, state, search, scraper)


# --- Chat loop ---------------------------------------------------------------


def display_qa_answer(answer: str, follow_ups: list[str]) -> None:
    console.print("\n")
    console.print(Panel(Markdown(answer), title="[bold blue]Answer[/bold blue]", border_style="blue", padding=(1, 2)))

    if follow_ups:
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, title="[bold]Follow-up Questions[/bold]")
        table.add_column("№", style="dim", width=4)
        table.add_column("Question", style="green")
        for i, q in enumerate(follow_ups, 1):
            table.add_row(str(i), q)
        console.print("\n")
        console.print(table)


def answer_from_state(question: str, state: ResearchState) -> None:
    """Q&A pass against the current ResearchState. Used on follow-ups."""
    result = _run_agent(qa_agent, QAInput(question=question), state)
    display_qa_answer(result.answer, result.follow_up_questions)


def research_follow_up(question: str, state: ResearchState, search: SearXNGSearchTool, scraper: WebpageScraperTool) -> None:
    """Follow-up that needs new material: plan up to ``FOLLOW_UP_SUB_TOPICS`` new sub-topics, research them, then QA.

    The planner sees the existing ``ResearchState`` via its context provider
    and is expected to propose only angles not yet covered. If it returns
    zero new sub-topics we print a visible warning so the user knows the
    QA answer rests on existing material, not new research.
    """
    state.question = question  # the planner / providers read the live question

    console.rule("[bold cyan]Extending research")
    new_sub_topics = plan_research(state, num_sub_topics=FOLLOW_UP_SUB_TOPICS)
    if not new_sub_topics:
        console.print("[yellow]Planner returned no new sub-topics — answering from existing state.[/yellow]")
    for sub_topic in new_sub_topics:
        research_sub_topic(sub_topic, state, search, scraper)

    answer_from_state(question, state)


def handle_follow_up(user_message: str, state: ResearchState, search: SearXNGSearchTool, scraper: WebpageScraperTool) -> None:
    """Route a single follow-up turn through decider → either research+QA or QA alone."""
    if state.agent_calls >= ResearchBudget.hard_call_cap:
        console.print("[red]Hard call cap reached — cannot process follow-up.[/red]")
        return

    decision = _run_agent(decider_agent, DeciderInput(user_message=user_message), state)

    title = "Performing new research" if decision.needs_research else "Answering from existing state"
    border = "yellow" if decision.needs_research else "green"
    console.print("\n")
    console.print(Panel(decision.reasoning, title=f"[bold {border}]{title}[/bold {border}]", border_style=border))

    if decision.needs_research:
        research_follow_up(user_message, state, search, scraper)
    else:
        answer_from_state(user_message, state)


def chat_loop() -> None:
    """REPL wrapper around the pipeline.

    First turn runs the full plan → research → write pipeline and prints the
    report. Every turn after that hands off to the decider, which routes to
    either another research pass (plan new sub-topics, research them, then QA)
    or straight to QA against the accumulated state. Type /exit to quit.
    """
    state = ResearchState(question="")
    wire_context_providers(state)
    search, scraper = _build_tools()

    console.print(Panel.fit("[bold blue]Deep Research — chat mode[/bold blue]\nType /exit to quit.", border_style="blue"))

    first_turn = True
    while True:
        prompt = "[bold blue]Your question:[/bold blue] " if first_turn else "[bold blue]Follow-up:[/bold blue] "
        try:
            user_message = console.input("\n" + prompt).strip()
        except (KeyboardInterrupt, EOFError):
            # Clean exit on Ctrl+C / Ctrl+D instead of a Rich traceback.
            console.print("\n[bold]Goodbye.[/bold]")
            return

        if not user_message:
            continue
        if user_message.lower() in ("/exit", "/quit"):
            console.print("\n[bold]Goodbye.[/bold]")
            return

        # Keep the REPL alive on turn-level failures (malformed structured
        # output, transient tool errors, etc.) instead of dropping the user's
        # accumulated ResearchState.
        try:
            if first_turn:
                first_turn = False
                run_initial_pipeline(user_message, state, search, scraper)
            else:
                handle_follow_up(user_message, state, search, scraper)
        except KeyboardInterrupt:
            _safe_print("Interrupted — returning to prompt.", style="yellow")
        except Exception as exc:
            _safe_print(f"Turn failed: {exc.__class__.__name__}: {exc}", style="red")
            _safe_print("Accumulated research state is preserved; try a different question.", style="dim")


# --- Internals ---------------------------------------------------------------


def _safe_print(message: str, style: str = "") -> None:
    """Print an error/status message without risking a recursive Rich failure.

    The chat loop's error handler must not itself raise — if Rich's own render
    path is what failed (e.g. a Windows encoding error), falling back to a
    plain builtin ``print`` keeps the REPL alive.
    """
    try:
        console.print(f"\n[{style}]{message}[/{style}]" if style else f"\n{message}")
    except Exception:
        try:
            print(f"\n{message}", flush=True)
        except Exception:
            pass


def _build_tools() -> tuple[SearXNGSearchTool, WebpageScraperTool]:
    search = SearXNGSearchTool(
        SearXNGSearchToolConfig(
            base_url=ChatConfig.searxng_base_url,
            max_results=ResearchBudget.search_results_per_query,
        )
    )
    scraper = WebpageScraperTool()
    return search, scraper


def _print_stats(state: ResearchState) -> None:
    console.print(
        f"\n[dim]Stats: {state.agent_calls} agent calls, {len(state.sources)} sources, "
        f"{len(state.learnings)} learnings.[/dim]"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        run(" ".join(args))
    else:
        chat_loop()
