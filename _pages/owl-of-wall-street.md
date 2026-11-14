---
layout: page
title: Owl of Wall Street
eyebrow: Building in public
permalink: /owl-of-wall-street/
description: "An investment company with one human and a team of AI agents. This page is the living record of what it is, what has been decided and how it is being built."
---

Owl of Wall Street is an investment research and advisory company that I run alone. Every employee is an AI agent: analysts, an investment committee, editors, a fact-checker, an educator, an engineer. I set the direction and approve what gets published. The agents do the work.

It has two goals, and they carry equal weight:

1. **Build a product that is useful** to people starting or continuing their investment journey.
2. **Show, in public, how to build a tech product from scratch with AI.** Every decision, agent design, failure and cost gets written up here.

<p class="notice">Status: the research and editorial agents are built and the weekly insight pipeline has had its first dry run. Nothing is published yet. This page is updated as the project moves. The <a href="#the-series">series below</a> covers each step in depth.</p>

## What it will offer

Everything is aimed at beginner and intermediate investors, anywhere in the world.

| Product | What it is | Price |
|---|---|---|
| **Learn** | A curriculum of educational articles, from compounding to portfolio construction | Free, always |
| **Insights** | One themed market insight a week, with a house thesis that can be proven wrong | Free, always |
| **Track Record** | A public record of every thesis and how it turned out, misses included | Free, always |
| **AI advisor** | A personal advisor on Telegram that turns your goals into a named ETF portfolio | Free to start; donations or a premium tier later |

Every article will come with a short animated video: the week's market moves in a minute, or a concept like compounding made visual. Every number in a video comes from the same data as the article and is fact-checked the same way.

## The advisor

The advisor is the main product. It is openly an AI agent, not a licensed adviser, and nothing it says is personal financial advice. It never places trades; you implement its suggestions with any broker you like.

It follows **goals-based wealth management**. Most advice starts from a risk score and hands you one portfolio. Goals-based planning starts from what you want the money for, a house deposit in six years or retirement in thirty, and asks one question per goal: which portfolio gives the best chance of getting there, within the risk you can live with?

- **Each goal gets its own target, date and priority.** Essential goals need a high chance of success; aspirational ones can take more risk.
- **Portfolios come from a fixed set of model portfolios** built from ETFs and reviewed every quarter. Each set is versioned, with a published changelog.
- **The fund universe follows where you live.** EU residents get UCITS ETFs; US residents get US-listed ETFs. The asset mix is the same, the funds differ.
- **Code does the maths, the model explains it.** Probabilities and allocations come from a deterministic engine. The language model never invents a number.
- **Stock picking is allowed, within limits.** It lives in a small satellite next to the ETF core, so the plan stays intact.

## How the company runs

{% include owl-org.html %}

The staff are **Claude Code subagents**: one file per role, each with its own instructions, tools and model. A subagent cannot call another subagent, so nobody manages anybody. Instead, the main Claude Code session plays **Chief of Staff**: it runs written workflows that call the agents in order and stop where I need to decide.

Agents hand work to each other through files in the repository, never through hidden context. Every brief, draft, red-team report and fact-check is a versioned file, so every decision can be traced. The full design is in [Designing a company where every employee is a Claude Code subagent]({% post_url 2026-11-08-Owl-Of-Wall-Street-Org-Design %}).

## The weekly insight

{% include owl-weekly-v2.html %}

I decide twice: which theme to write about, and whether the finished piece ships. Before I pick, the Risk Officer screens the three candidate themes, so weak ideas are caught before anyone drafts them. Everything between the two decisions is delegated, and every review loop has a cap. When a loop hits its cap, the work comes back to me instead of being forced through. The first dry run, and what it changed, is in [The first dry run of an AI investment team]({% post_url 2026-11-14-Owl-Of-Wall-Street-First-Dry-Run %}).

## Decisions so far

| Decision | Choice |
|---|---|
| Audience | Global, beginner and intermediate investors |
| Execution | None. Users implement portfolios with their own broker |
| Advice style | Named ETF portfolios, with tickers and weights |
| Fund coverage | US-listed and EU UCITS ETFs, picked by where the user lives |
| Methodology | Goals-based wealth management |
| Website content | Always free |
| Advisor monetisation | Donations, freemium, or both. Payments get built; switching them on is a later decision |
| Positioning | The advisors are clearly AI agents and are used at the user's own risk |
| Internal agents | Claude Code subagents, shown by role only |
| User-facing agents | Application code served by our API. Only the advisor has a name |
| Model portfolios | Re-evaluated every quarter and when a fund changes; versioned |
| Repository | Monorepo |
| Rollout | Workflows run by hand first, automated after about four clean weeks |
| Agent knowledge | Analytical skills from InvestPal, distilled from investing books, adapted to ETF exposures |
| House theses | About an exposure, and may name the funds that deliver it. Companies are discussed as evidence, not as the subject |
| Compliance | Clear "not investment advice" disclosures. Legal questions are noted, never a gate, at this stage |
| Data | Web search until Owl builds its own data tools. No paid data licences |
| Thesis outcomes | Every thesis states what would prove it wrong. No mechanical scoring; how outcomes are judged is still open |

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0. Define | Product definition, org design, data sources | Done |
| 1. Content engine | Research and editorial agents, website, first educational articles, weekly insights | **In progress**: agents built, first dry run done |
| 2. Planning engine | ETF universes, model portfolios, goals-based probability engine | |
| 3. Advisor beta | Telegram bot, onboarding, goals, memory; closed beta | |
| 4. Public launch | Track Record, The Team page, public advisor, payments behind a switch | |

## The series

{%- assign series = site.posts | where: "series", "owl-of-wall-street" | sort: "date" %}
<ul class="posts">
  {%- for post in series %}
  <li class="post">
    <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
    <span class="mono date">{{ post.date | date: "%-d %b %Y" }}</span>
    <p>{{ post.summary | default: post.description }}</p>
  </li>
  {%- endfor %}
</ul>

## How this page was made

The product definition came out of a two-day working session with Claude Code. I brought the idea; Claude proposed structure, pushed back where it saw risk, and kept a decisions log in the repository as we went. This page is a summary of that log. The process of building the company with AI is part of what the company demonstrates.

---

*Owl of Wall Street's advisors are AI agents, not licensed financial advisers. Nothing here or produced by the project is personal financial advice.*
