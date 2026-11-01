---
layout: post
title: "Building an AI investment analyst that follows a method"
date: 2026-11-01
tags: [investing, AI, MCP, agents]
description: "How InvestPal is built: written analysis procedures the agent must follow, five small services on your own machine, memory as structured state, and brokerage access designed around real money."
featured: true
short_title: "Building an AI investment analyst that follows a method"
summary: "How InvestPal is built: written analysis procedures, five local services joined by MCP, memory as structured state, and brokerage access designed around real money."
---

Ask a general-purpose AI whether a stock is worth buying and you get a fluent answer put together on the spot. It reads well. It is also different every time you ask, and nothing in it tells you how the conclusion was reached or what would change it.

[InvestPal](/investpal/) is built around the opposite idea: an analyst should follow a method. It works through written, step-by-step procedures for each kind of question, it runs on your own machine, and it plugs into the AI tools you already use. This post is about how it is built and why it is built that way. The [product page](/investpal/) covers what it does, and the code is [on GitHub](https://github.com/OrestisStefanou/InvestPalEcosystem).

Three decisions shape everything below:

- **Method over improvisation.** The agent reaches for a written procedure before it reasons freehand.
- **Local-first.** No hosted backend, no account, no tenancy. Every service and all of your data stay on your machine; the only thing that leaves it is what the model you choose needs to see to answer.
- **Client-agnostic.** The backend speaks the Model Context Protocol (MCP), so any capable assistant can become the front-end.

## The shape of it

InvestPal is five small services running on localhost, plus one database file.

<figure class="diagram">
<svg viewBox="0 0 720 340" role="img" aria-label="Four client surfaces on the left: Claude Code, Claude Desktop, any MCP client, and the web app or your own app over REST. They connect to the InvestPal core in the middle, which exposes REST on port 8000 and MCP on port 9000 and stores everything in one local Turso/SQLite file. The core connects over MCP to four servers on the right: Market Data in Go, and Alpaca, Coinbase and Interactive Brokers in Python. Interactive Brokers talks to its own Java gateway. Claude Code and Claude Desktop also connect to the four servers directly.">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dg-head"/></marker></defs>
<text class="dg-lbl" x="10" y="14">CLIENTS</text>
<text class="dg-lbl" x="560" y="14">MCP SERVERS</text>
<path class="dg-line dg-dash" d="M90,56 V32 H630 V54" marker-end="url(#ah)"/>
<text class="dg-sm" x="360" y="27" text-anchor="middle">Claude Code and Claude Desktop also connect directly</text>
<rect class="dg-box" x="10" y="56" width="160" height="40" rx="6"/><text x="22" y="81">Claude Code cockpit</text>
<rect class="dg-box" x="10" y="112" width="160" height="40" rx="6"/><text x="22" y="137">Claude Desktop</text>
<rect class="dg-box" x="10" y="168" width="160" height="40" rx="6"/><text x="22" y="193">Any MCP client</text>
<rect class="dg-box" x="10" y="224" width="160" height="40" rx="6"/><text x="22" y="249">Web app, your own app</text>
<line class="dg-line" x1="170" y1="76" x2="266" y2="130" marker-end="url(#ah)"/>
<line class="dg-line" x1="170" y1="132" x2="266" y2="146" marker-end="url(#ah)"/>
<line class="dg-line" x1="170" y1="188" x2="266" y2="162" marker-end="url(#ah)"/>
<line class="dg-line" x1="170" y1="244" x2="266" y2="178" marker-end="url(#ah)"/>
<text class="dg-sm" x="190" y="100">MCP</text>
<text class="dg-sm" x="190" y="232">REST</text>
<rect class="dg-core" x="270" y="110" width="190" height="96" rx="8"/>
<text class="dg-b" x="284" y="134">InvestPal core</text>
<text class="dg-sm" x="284" y="154">Python, FastAPI</text>
<text class="dg-sm" x="284" y="172">REST :8000 · MCP :9000</text>
<text class="dg-sm" x="284" y="190">3 agents · 15 skills</text>
<line class="dg-line" x1="365" y1="206" x2="365" y2="252" marker-end="url(#ah)" marker-start="url(#ah)"/>
<rect class="dg-box" x="290" y="256" width="150" height="48" rx="6"/>
<text x="304" y="277">investpal.db</text>
<text class="dg-sm" x="304" y="295">local Turso / SQLite</text>
<line class="dg-line" x1="460" y1="140" x2="546" y2="76" marker-end="url(#ah)"/>
<line class="dg-line" x1="460" y1="150" x2="546" y2="132" marker-end="url(#ah)"/>
<line class="dg-line" x1="460" y1="166" x2="546" y2="188" marker-end="url(#ah)"/>
<line class="dg-line" x1="460" y1="176" x2="546" y2="244" marker-end="url(#ah)"/>
<text class="dg-sm" x="492" y="100">MCP</text>
<rect class="dg-box" x="550" y="56" width="160" height="40" rx="6"/><text x="562" y="81">Market Data <tspan class="dg-sm">Go</tspan></text>
<rect class="dg-box" x="550" y="112" width="160" height="40" rx="6"/><text x="562" y="137">Alpaca <tspan class="dg-sm">Python</tspan></text>
<rect class="dg-box" x="550" y="168" width="160" height="40" rx="6"/><text x="562" y="193">Coinbase <tspan class="dg-sm">Python</tspan></text>
<rect class="dg-box" x="550" y="224" width="160" height="40" rx="6"/><text x="562" y="249">Interactive Brokers</text>
<line class="dg-line" x1="630" y1="264" x2="630" y2="286" marker-end="url(#ah)"/>
<rect class="dg-box dg-faint" x="550" y="290" width="160" height="40" rx="6"/><text class="dg-sm" x="562" y="315">IB gateway, Java</text>
</svg>
<figcaption>Every service listens on localhost. The brokerage servers are optional; InvestPal works as a research tool on market data alone.</figcaption>
</figure>

**The core** is a Python FastAPI service. It exposes the same capabilities twice: as a REST API for ordinary applications, and as an MCP server for AI clients. It owns the agents, the analysis procedures and all of the user's state.

**The data and brokerage servers** are separate MCP servers, one per source: a Go server for market data, and Python servers for Alpaca, Coinbase and Interactive Brokers. Each one is small, does one thing, and can be used on its own by any MCP client.

**State** lives in one Turso/SQLite file, created on first start. There is no database server to run. Syncing that file to Turso Cloud is possible, and deliberately manual: nothing syncs on startup, on shutdown or on a timer, so a second machine never overwrites the first by surprise.

The detail that matters most is the dotted line. Claude Code and Claude Desktop are already capable agents, so they connect to the market data and brokerage servers **directly**, as native MCP clients. The core only coordinates those servers on behalf of clients that cannot speak MCP themselves, such as the web app. MCP is the seam that lets one backend serve very different front-ends without a translation layer for each.

## Skills: written procedures instead of improvisation

The heart of InvestPal is fifteen **skills**. A skill is a written analytical procedure: what to look at, in what order, which numbers to compute, how to interpret them, and when the method does not apply. Their lineage is explicit. Valuation and asset-based reasoning come from Graham and Dodd's *Security Analysis*; cycles, sentiment and second-level thinking come from Howard Marks' *The Most Important Thing*.

They are grouped by the question an investor is actually asking:

| Question | Skills |
|---|---|
| Is this a good business? | Competitive moat · Earnings quality · Cash generation · Balance sheet strength · Profitability and pricing power · What management is signalling |
| What is it worth? | Intrinsic value · Price against value · Against its peers |
| Should I act now? | Second-level thinking · Market temperature · Macro conditions · Testing a theme |
| How exposed am I? | Portfolio risk · Margin of safety |

### How the agent reaches them

Skills are not pasted into the system prompt. Fifteen full procedures would crowd out the conversation, and most questions need one or two of them. Instead the core exposes two tools:

- **List skills** returns every skill with a one-line summary of what it is for and when to use it.
- **Get skill** returns the full written procedure for one of them.

The agent's instructions are blunt about using them: always prefer a skill over ad-hoc analysis, and skipping a relevant skill is a defect. So a question like "does this company have a moat?" turns into a short, predictable sequence: list the skills, fetch the moat procedure, gather the data it asks for, and work through it step by step.

### What a procedure looks like

The competitive moat skill is a good example of why this matters. A freehand answer to "does this company have a moat?" usually lists some reassuring adjectives: strong brand, loyal customers, network effects. The procedure asks for numbers instead.

It starts by comparing two independent estimates of value. The first is what it would cost to rebuild the company's assets from scratch. The second is its **earnings power**: what its current, sustainable earnings are worth if they never grow. If earnings power is well above the cost of rebuilding the assets, competitors should have rushed in and competed the excess away. If they have not, something is stopping them, and that something is the moat. If the two values are roughly equal, the company is a commodity business, however good its brand looks.

Only then does the procedure look for the cause: customer captivity, cost advantages, economies of scale, network effects. It ends with an estimate of how long the advantage is likely to last. The answer comes out the same way every time, and you can check every step of it.

### Why written procedures

- **Consistency.** The same question gets the same method, so two answers about two companies can actually be compared.
- **Auditability.** You can read the procedure the agent followed, disagree with a step, and see exactly where your view differs.
- **Versioning.** Procedures are code. They live in the repository, change through commits, and every change is reviewable.
- **Model independence.** The method lives outside the model. Swap the model and the analysis still follows the same steps.

## Three agents with separate jobs

Inside the core, three agents share the same tools but have different jobs. Each is a LangChain agent that reaches its tools over MCP.

| Agent | Job |
|---|---|
| Investment manager | The advisor you talk to. Holds the persona, reaches for skills, calls the data and brokerage tools. |
| Memory manager | Runs in the background on a cheaper model. Reads the conversation and maintains your long-term memory, so the advisor never stops mid-answer to take notes. |
| Workflow executor | Has no conversation at all. Runs scheduled work when it comes due and stores the report. |

Keeping memory out of the conversation lets the advisor stay attentive without being slow. Note-taking is real work: deciding what changed, superseding old facts, filing things in the right place. Doing it inline either delays the answer or gets skipped. A background agent on a cheaper model does it properly and costs little.

The model and provider are configured per agent, and Anthropic, OpenAI and Google models are all supported. In the Claude Code cockpit, Claude Code itself is the model, so InvestPal's own agent and its API key are not involved at all.

## Memory as structured state

The easy way to give an agent memory is a pile of notes and a vector search. InvestPal keeps notes like that, but most of what matters to an investor has a shape, so most of its memory is structured.

| What | How it is kept |
|---|---|
| Your profile | Goals, risk tolerance, horizon and constraints, as notes that are superseded rather than deleted, so the history of how your situation changed is kept |
| Holdings | What you own, from a broker or recorded by hand. Every row says where it came from and when it was true. No prices and no P&L are stored, because they go stale immediately |
| Ticker records | One per name you care about: the thesis, the entry trigger, the falsifier (what would prove the thesis wrong), and whether it is watched, held, exited or rejected |
| Conversation notes | Free-form notes from past conversations, searchable by meaning |
| Reminders and workflows | Action items the advisor is tracking, and scheduled work with its reports |

The ticker record does the most work. Writing down the falsifier when you buy is the cheapest protection there is against falling in love with a position. When the advisor revisits a holding, it checks the current facts against the conditions you wrote down at the time, not against whatever the story has become since.

Semantic search over conversation notes uses embeddings computed **locally**, on the CPU, with a small open embedding model. The text of your notes is never sent anywhere to be embedded, which matters for something as personal as your finances.

## Work that runs on a schedule

An analyst is useful partly because they keep watching after you leave the room. InvestPal does this with **workflows**: a name, a description of what to achieve, and a cron schedule. "Review my portfolio every Monday morning and tell me anything that changed" is one workflow.

When a workflow comes due, the workflow executor runs it with the same tools and skills as the advisor and stores the report, so it is waiting for you the next time you open InvestPal. A workflow is marked as running while a run is in flight, which works as a lock: two triggers firing close together cannot run the same review twice.

Each front-end picks up these reports where it makes sense. In the Claude Code cockpit, a session-start hook loads your profile and open reminders and runs any workflows that came due while you were away. The web app shows each workflow's history of reports.

## Brokerage access with real money attached

Connecting an AI to a brokerage account is the part where a design mistake costs money, so the rules here are strict and simple.

**A credential lives in exactly one process.** Brokerage keys sit in a separate secrets file with owner-only permissions. At startup, each key is handed to its own brokerage server and to nothing else. The keys never travel over an MCP connection and never pass through the model's context. The Claude Code cockpit is explicitly denied access to the secrets file.

**Read-only removes the order tools entirely.** Each brokerage has a read-only switch. When it is on, the tools that place orders are not just refused at call time; they are never registered, so the agent cannot see them, ask for them or be talked into calling them.

**Defaults are the safe option.** Alpaca points at paper trading unless you explicitly configure the live endpoint. Interactive Brokers never accepts a trading warning on your behalf: an order that triggers a warning comes back as needing confirmation, with the warning text, and stays off the market until a separate confirm call.

None of this is needed to use InvestPal. Without any brokerage connected, it is a complete research tool, and you can record holdings by hand.

## Data without a vendor

The market data server draws only on free, public sources and needs no API keys:

| Source | Used for |
|---|---|
| FRED | Economic indicators and commodities |
| ECB reference rates | Currency exchange rates |
| SEC EDGAR | Insider transactions |
| stockanalysis.com | Stocks, ETFs, sectors, financials and news |
| dataroma | What well-known value investors hold |
| CoinGecko | Cryptocurrency data |
| CoinDesk and CoinTelegraph | Crypto news |
| Polymarket | Prediction market odds |

The trade-off is that free sources come with their own rules. Some have rate limits, and the SEC expects automated clients to identify themselves with a real contact address. Keeping every source behind one MCP server means those details live in one place, and swapping a source does not touch the agents or the skills.

## One backend, many front-ends

Because the core speaks both MCP and REST, the same InvestPal can be used in several ways:

| Front-end | Best for |
|---|---|
| Claude Code cockpit | Daily use. Run Claude Code inside the InvestPal folder and it becomes the advisor: it loads the persona, your profile and reminders, and runs due workflows |
| Claude Desktop | Occasional questions in an ordinary chat, no terminal needed |
| Any MCP client | Bringing InvestPal into whatever assistant you already use |
| Local web app | A browser view of conversations, workflows and their reports, and reminders |
| REST API | Building your own interface, or wiring InvestPal into something you already run |

None of them needs anything hosted. A single `make setup` installs the services, writes the configuration and starts the stack, and `make doctor` tells you what is missing.

## Wrapping up

The goal behind every decision here is the same: an AI analyst you can trust with questions about your money. You should be able to see how it reached an answer, your records should stay on your own machine, and it should not be able to act on your accounts unless you deliberately allow it.

Everything is open source. Start with the [InvestPal page](/investpal/) for what it does, or go straight to the [code on GitHub](https://github.com/OrestisStefanou/InvestPalEcosystem).
