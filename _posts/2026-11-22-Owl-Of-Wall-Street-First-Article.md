---
layout: post
title: "Two failed runs and a first article: what an AI investment team needed"
date: 2026-11-22
tags: [AI, agents, Claude Code, Owl of Wall Street]
series: owl-of-wall-street
description: "Owl of Wall Street's AI agents failed a second dry run, then wrote their first approved article, on the Anthropic IPO. The fix was not a better agent but a different job. What the reviews caught, including omissions that flattered the company whose models run every agent."
short_title: "Two failed runs and a first article"
summary: "The second dry run was rejected too. Then I realised I had asked the agents for the wrong thing. What changed, how the first approved article was made, and what the reviews caught."
---

The [last post]({% post_url 2026-11-14-Owl-Of-Wall-Street-First-Dry-Run %}) ended with a plan: re-run the same week with the fixes, then do a real run aimed at publishing. Both happened. The re-run failed too. In between, I worked out that the problem was not the agents. I had given them the wrong job.

This post covers the second failure, the change it led to, how the first approved article was made, and what the review loops caught along the way.

## The second dry run failed earlier, which was the point

The re-run used the same week, 9 to 13 November, with the workflow changes from the first run. The CIO proposed a thesis on US inflation-protected government bonds (TIPS): over 12 months, a 2036 TIPS issue would beat the matching ordinary Treasury.

Two things happened before any drafting was wasted.

**The theme screen worked.** The Risk Officer now screens the three candidate themes before I pick one. It flagged the weak point of the TIPS theme straight away: the thesis depended on what markets expected inflation to be over the next year, and no free source gave that figure. The final verdict came back to exactly that point.

**There was no fund for the exposure.** No ETF held only 7 to 10 year TIPS. The closest fund had 22% of its money in that range. That raised a question I hadn't asked: why must every thesis use an ETF? It doesn't. ETFs stay the main focus of Owl's portfolios, but a thesis may now name a bond held directly, or a single company against a benchmark.

The red team then rejected the draft in round 2. Its reasoning: the house view was worth about 3 to 4 basis points of inflation pricing, against a 14 bps move in the previous two weeks. The CIO agreed. I skipped the week. Run 2 cost about two thirds of run 1 and stopped before any editing, which is what moving the checks earlier was meant to do.

## The real problem: I had asked for an edge

Two runs had now spent their effort hunting for a few basis points of advantage in US government bonds. Reading the second rejection, I saw why. The pipeline was built to beat the market: each thesis had to name a price, explain why the market was wrong, and be scored right or wrong against it. That is what a hedge fund does. It is not what Owl is for.

Owl's readers are beginner and intermediate investors with goals: a pension, a house, a child's education. What they need each week is plain: what happened, why it matters, and what it means for someone investing for the long term. Often the honest answer is "nothing changes, keep going".

So I rewrote the job:

| Before | After |
|---|---|
| The thesis is a bet against the market price | The thesis is the house view on what the week means for a long-term investor |
| A week without an edge is skipped | Every week publishes; "no change" is a valid view |
| Each thesis is right or wrong on one number | Each view has a review date and what would change it, and is marked held or changed when revisited |
| The red team asks "why is the house more right than the market?" | The red team asks "is this sound, sourced, balanced and useful to a beginner?" |

None of the agents' skills changed. Second-level thinking still asks what is already priced; it now guards readers against chasing a story instead of hunting for an edge.

## The first real run: the Anthropic IPO

The week of 16 to 20 November was easy to pick. Anthropic listed on Nasdaq on Thursday, a record IPO, in a week when the S&P 500 fell 8,61% into a correction. I gave the pipeline a steer: make the IPO a candidate theme, and ask what it means for the markets. Everyone holding a Nasdaq-100 or world index fund was about to own Anthropic without choosing to.

There was an obvious awkwardness. Every Owl agent runs on Claude, which Anthropic makes. My view was that this isn't a real conflict, because the article makes no call to buy or sell the shares. But readers should know, so the article says it near the top. I also asked the red team to look specifically for a tilt in Anthropic's favour.

| Step | Agent | What happened |
|---|---|---|
| Research | Market Analyst, Macro Economist | Two briefs. The Market Analyst tested a theory that crypto fell because investors sold to buy Anthropic: the slide started in late October, and no flow data linked the two. Called unproven |
| Themes and screen | CIO, Risk Officer | Three themes. Both recommended the IPO, framed as what it means for index-fund holders |
| **Gate 1** | Me | Picked it |
| Funds | ETF Strategist | Example funds for Nasdaq-100 and world-index holders. Could not read the prospectus (below), so Anthropic's index weights were estimated |
| **Decision** | Me | Draft with the estimates, clearly labelled |
| Draft and red team | CIO, Risk Officer | Round 1 `revise`, round 2 `approve-with-fixes` |
| Edit | Editor | 2.065 words, then 1.912 |
| Fact-check | Fact-Checker | Round 1 `fail` (62 verified, 7 corrected); round 2 `fail` on one figure |
| **Gate 2** | Me | Review, changes, approval, merge |

The pipeline itself took about 1,5 hours: 23 agent runs and about 1,46 million tokens. My review the next day added 10 more runs.

## What the reviews caught

### Omissions that flattered Anthropic

The red team found what I asked it to look for. Four omissions in the first draft all leaned the same way. It gave the share price only against the IPO price, not against the much higher first trade. It gave no valuation. It left out a $26bn sale of a separate share class to chipmakers. And it understated how many insider shares would come to market in the next six months.

None of these was false. Together they made the story rosier than the facts. Its main point went further: "changes little" was true on the day Anthropic entered the index, but not over the six months the view would be tracked. As insiders become free to sell, Anthropic could become a 3% to 4% holding in a Nasdaq-100 fund, tied to the same chipmakers the fund already holds. The view changed to say so: it enters small and is likely to grow.

I can't tell whether a model built by Anthropic leans towards Anthropic, and this run doesn't prove it either way. What it shows is that a reviewer told to look for a specific bias found one, and that the fix was cheap.

### A wrong price, and what it touched

The fact-checker found that Anthropic's Friday close was $88,91, not $88,30: one secondary source disagreed with three others. That price fed the index-weight estimates, so the fix went through three agents in order: fund data, then the draft, then the article. In round 2, the fact-checker found that one estimate was still wrong, because at the corrected price Anthropic had moved past a neighbouring company in the ranking. That was the second `fail`, which by the rules comes to me. I chose to round the figure and skip a third round.

### The prospectus nobody could read

The prospectus, the company's own filing, was 14 MB. The agents' web tool stops at 10 MB, so five agents in a row failed to read it and fell back on press summaries. The article carried a line saying "we could not read the whole prospectus", which I didn't like.

The fix was trivial: I downloaded it, and it was converted to text the agents could search section by section. Reading the filing changed three things. The lock-up amounts were slightly different from the press summary (303 million shares instead of 299 million, and so on). One sentence was wrong: it said free float could pass a third of the listed shares by early February, and the filed numbers show it stays just under until after the 2026 results. And one correction turned out to be wrong. Claude, running the workflow, told me the filing contradicted a share count we had used; the ETF Strategist showed that the filing contains both numbers, before and after a private placement, and our figure was right. A third, narrow fact-check verified all 30 changed figures against the filing.

## The article

The headline I approved: *Anthropic IPO and what it means for investors.* The view, as it appears in the article:

> **Our view:** Anthropic joining index funds changes little for most long-term investors. In a world index fund we estimate it enters at about 0,1%, and it would stay under 1% even if all its listed shares counted. In a Nasdaq-100 fund we estimate it enters at about 1,3% to 1,9%, and its weight is likely to step up through spring 2027 as insiders become free to sell, possibly to around 3,7% at today's price, which would make it a top-10 holding.
>
> **What it means in practice:** For most long-term investors, no change: keep contributing and rebalance as planned. There is no need to buy ANT or an AI fund to "get" Anthropic; index holders will own it anyway.
>
> **We'll revisit this by:** 21 May 2027.

What I liked most was not the view but the writing. The article explains how an index decides a company's weight, what free float and lock-ups are, and why a fund's concentration matters, without a word of jargon left unexplained. That became a company principle.

## What I changed

| Change | Why |
|---|---|
| Plain language is a principle: use a financial term when it is the right tool, explain it where it appears and why it matters, never trade accuracy for simplicity | It's what made the article good, and it serves the education goal |
| Article length is a guide, not a limit | A hard 1.500-word cap forced cuts the topic didn't need and added a loop sending the article back for length alone |
| The AI disclosure names the models, and an article about Anthropic says so at the top | Readers should know what wrote the article |
| Approval flow: the pipeline opens a draft PR; when I approve, the Chief of Staff removes the draft label and opens the view in the track record; I merge | "Merge equals publish" meant nothing without a website |
| No holdings disclosure for now | My call at this stage; to revisit when articles reach readers |
| A CTO agent | The data gaps now need an owner |

The CTO is a different kind of agent. The others are called by the weekly workflow. The CTO is one I talk to directly, as a main session started with `claude --agent cto`. It proposes options with a recommendation, records each technical decision in a plain-language decision record, and delegates the work: one kind of subagent to read code, another to write it, each writer on its own copy of the repository. It reviews every diff before I see it.

## What the data layer needs now

Three runs have logged their data gaps, and the same ones keep coming back:

1. **Reading large documents locally.** Filings that are too big for the web tool, downloaded once and searched by section.
2. **Daily closes from one reliable source** for indices, stocks, currencies, commodities and crypto. A single wrong close caused this week's first fail.
3. **Index data**: which companies are in an index, at what weight, and the providers' own announcements.
4. **Bond prices** for specific issues, which the first two runs needed and could not get.

That list is the CTO's first job, together with the question this run made obvious: articles are approved, but nobody can read them yet. Where and how they get published is the next decision.

## The numbers

| | Run 1 | Run 2 | W47 |
|---|---|---|---|
| Agent runs | 18 | 12 | 23, plus 10 in my review |
| Tokens | 1,51m | 1,06m | 1,46m, plus 0,48m in review |
| Red team | `revise`, `approve` | `revise`, `reject` | `revise`, `approve-with-fixes` |
| Fact-check | `fail`, `fail` | not reached | `fail`, `fail`, then `pass` after review |
| Outcome | Closed at the review limit | Week skipped | Approved and merged |

Run 2's token figure overstates its usage slightly: resumed agents reported cumulative totals, which W47 corrected.

## Next

Next week's article reports the index weights MSCI and Nasdaq actually publish against our estimates, and says plainly if they fall outside them. Then the CTO's first session. The [project page](/owl-of-wall-street/) lists the series as it grows.

---

*Owl of Wall Street's analysts are AI agents, built on Anthropic's Claude models, and not licensed financial advisers. The view quoted in this post is an example of the process, not investment advice or a recommendation to buy or sell any security.*
