---
layout: post
title: "Designing a company where every employee is a Claude Code subagent"
date: 2026-11-08
tags: [AI, agents, Claude Code, Owl of Wall Street]
series: owl-of-wall-street
description: "The org design of Owl of Wall Street, an investment company with one human and nine AI agents: why nobody manages anybody, why the filesystem is the office, and where the human decides."
featured: true
short_title: "A company where every employee is a Claude Code subagent"
summary: "The org design of Owl of Wall Street: nine Claude Code subagents, no AI managers, files as the hand-off, and two human gates."
---

I'm starting an investment research company with no employees. [Owl of Wall Street](/owl-of-wall-street/) will publish educational articles and a weekly market insight, and run a personal AI advisor on Telegram. I'm the only human. Everyone else is an AI agent.

That raises a design problem before it raises a coding problem: what does the org chart of a company like that look like? How many agents, doing what, reporting to whom, and where does a person still have to decide? This post is the answer I landed on, and the reasoning behind it. The [project page](/owl-of-wall-street/) covers the product itself.

## The constraint that shaped everything

The internal staff are [Claude Code](https://docs.claude.com/en/docs/claude-code) subagents. A subagent is a markdown file with a system prompt, a list of allowed tools and a model. Claude Code runs it in its own clean context and gets a result back.

One rule of that system decided the whole structure: **a subagent cannot start another subagent.**

My first sketch had a CIO who would "ask" the analysts for research, the way a human CIO would. That cannot work. So instead of fighting the rule, I built on it:

- **Nobody manages anybody.** No agent coordinates another agent.
- **Coordination lives in written workflows.** A workflow is a step-by-step procedure, stored in the repository, that the main Claude Code session executes. It calls agents in order or in parallel, passes them the right files, and stops where I need to decide.
- **The main session gets a role of its own: Chief of Staff.** It runs the procedures. It does not do the research or write the articles.

This turned out to be the better design anyway. A human CIO asking analysts for research produces an invisible, unrepeatable process. A workflow file is explicit, versioned and the same every week. When something goes wrong, I fix the procedure, not a conversation.

## Three building blocks

| Block | What it is | In a human company |
|---|---|---|
| **Subagents** | One file per role: instructions, allowed tools, model | The employees |
| **Skills** | Written procedures and reference material: house style, the thesis template, how to pick an ETF, how goals-based planning works | The handbook |
| **Workflows** | Slash commands such as `/weekly-insight` that the main session executes step by step | The operating procedures |

Agents are *who*, skills are *how*, workflows are *when and in what order*. Keeping the three apart means I can change a procedure without touching a role, or reuse one skill across several roles.

## The org chart

{% include owl-org.html %}

Nine subagents in five departments for version one:

| Agent | Job | Model |
|---|---|---|
| Market Analyst | What happened in markets this week, with numbers and sources | Sonnet |
| Macro Economist | The macro regime and its direction; the return and risk assumptions behind the model portfolios | Opus |
| ETF Strategist | The US and UCITS fund universes, the model portfolios, ETF examples for each thesis | Sonnet |
| CIO | Proposes three candidate themes, then writes the thesis and the draft | Opus |
| Risk Officer | Tries to prove the CIO wrong | Opus |
| Editor | Makes the article clear for beginners | Sonnet |
| Compliance & Fact-Checker | Traces every number and claim to a source | Sonnet |
| Educator | Writes the Learn curriculum | Sonnet |
| CTO | Owns the architecture and builds the product | Opus |

The heavier model goes where judgment matters most: forming a view, attacking it and setting long-term assumptions. These are starting points. The cost and quality of every run is logged, so the choices will move with evidence.

The internal agents are shown by role only. Only the user-facing advisor gets a name. A name on an analyst adds nothing a reader needs; a role tells you exactly what it is accountable for.

## Separation of powers

The most important design choice is not which agents exist, but **what each one is not allowed to do**.

- **Writers never approve their own work.** Every public piece passes at least one independent reviewer agent, then me.
- **The Risk Officer's only job is to disagree.** It writes the strongest counter-argument, lists missing risks, checks that the thesis states what would prove it wrong, and returns `approve` or `revise`.
- **The Editor cannot change claims or numbers.** It can make a sentence clearer. If it thinks a claim is wrong, it flags it back; it does not fix it quietly.
- **The Fact-Checker cannot edit the article at all.** It can only write its own report: every figure, its source, and whether it holds up. A checker that can edit is tempted to patch instead of report.
- **Numbers come from tools, never from memory.** Any figure in an article has to trace back to a data call or a file. That is what the Fact-Checker verifies.

These rules are enforced, not requested. Each agent's tool list is an allowlist, and hooks block writes outside the folders it owns.

## The filesystem is the office

Agents never pass work through conversation. Each step writes a file; the next step reads it.

```
company/
  research/2026-W45/
    market-brief.md        Market Analyst
    macro-brief.md         Macro Economist
  insights/2026-W45/
    themes.md              CIO: three candidates
    theme-decision.md      me
    draft.md               CIO
    redteam.md             Risk Officer
    edit-notes.md          Editor
    factcheck.md           Fact-Checker
  theses/registry.yaml     every thesis, scored over time
  portfolios/models/v1/    model portfolios, US and UCITS
```

Three things follow from this:

- **No hidden state.** Every subagent starts clean, so its inputs are exactly the files it was pointed to. If an output is wrong, I can see precisely what the agent saw.
- **Everything is auditable.** Git history is the company's paper trail: who wrote what, who objected and what changed.
- **The work documents itself.** The artifacts are the raw material for posts like this one.

## A week at the company

{% include owl-weekly.html %}

| Step | Who | Output |
|---|---|---|
| 0 | A script, no model | Open theses scored against market data |
| 1 | Market Analyst and Macro Economist, in parallel | Two briefs |
| 2 | CIO | Three candidate themes |
| 3 | **Me** | The theme to write about |
| 4 | CIO, with the ETF Strategist | Draft, plus a new entry in the thesis registry |
| 5 | Risk Officer | Red-team report; back to step 4 if `revise`, at most twice |
| 6 | Editor | Edited article and notes on what changed |
| 7 | Fact-Checker | Pass or fail; back to the responsible agent on fail, at most twice |
| 8 | Chief of Staff | A pull request with the article, every artifact and the run's cost |
| 9 | **Me** | Review and merge, which publishes it |

I decide twice: what to write about, and whether it ships. Every loop has a cap, and a loop that hits it hands the work back to me instead of forcing it through.

Every thesis is logged in a structured registry: the claim, the horizon, a measurable proxy and what would invalidate it. Step 0 scores the open ones every week, and the results go on a public track record, misses included.

## Two kinds of agents

Not every agent in the company is a subagent.

| Kind | Built as | Why |
|---|---|---|
| **Internal staff** | Claude Code subagents | The work is batch, weekly and gated by a human. Claude Code already gives me tools, permissions, hooks and a session to run it in |
| **The product** | Application code behind our own API | The advisor talks to many users at once, needs per-user memory, fast responses and its own guardrails. That is a service, not a session |

The two never call each other at runtime. The company publishes versioned files, such as model portfolios and articles, and the product reads them. If a weekly run fails, nobody's advisor notices.

## What I chose not to do

- **No CEO agent.** Direction and final approval stay with a person. That is the point of the experiment, not a gap in it.
- **No agents managing agents.** The platform rules it out, and workflows do the job better.
- **No automation on day one.** For the first four weeks or so I run each workflow by hand and watch every step. Prompts get tuned against real output, and only then does it go on a schedule.
- **No big team up front.** Engineering starts as a single CTO agent. It will grow into something like my [software factory](/software-factory/), but only when the work demands it.

## Principles

| Principle | In practice |
|---|---|
| One role, one job | Every agent has a single job description and a fixed set of outputs |
| Least privilege | Allowlisted tools; reviewers can only write their own report |
| Independent review | The writer never approves; the founder always does |
| Data from tools | No number without a traceable source |
| Bounded loops | At most two review rounds, then a human |
| Everything logged | Model, tokens, cost and duration for every agent run |

## Next

The next step is writing the agent definitions and running `/weekly-insight` by hand on a real week, without publishing anything. For that first run the agents use plain web search instead of a proper data layer. That is deliberate: the run will show exactly which data the agents reach for and where the Fact-Checker cannot trace a number, and the data layer gets built from that list rather than from a guess. Both steps get their own write-up; the [project page](/owl-of-wall-street/) lists the series as it grows.
