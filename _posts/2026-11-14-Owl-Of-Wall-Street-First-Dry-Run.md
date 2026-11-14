---
layout: post
title: "The first dry run of an AI investment team, and what it changed"
date: 2026-11-14
tags: [AI, agents, Claude Code, Owl of Wall Street]
series: owl-of-wall-street
description: "Owl of Wall Street's seven AI agents wrote their first weekly market insight end to end. The red team rebuilt the thesis, the fact-checker failed it twice, and the run changed the workflow, the rules and the data plan."
short_title: "The first dry run of an AI investment team"
summary: "Seven Claude Code subagents ran the weekly insight pipeline on a real week. What the skills from InvestPal added, what the review loops caught, what broke, and what I changed."
---

The [last post]({% post_url 2026-11-08-Owl-Of-Wall-Street-Org-Design %}) designed a company where every employee is a Claude Code subagent. This one is about the first time that company did its job: seven agents wrote a weekly market insight about a real week, from research to fact-check, with nothing published.

It did not produce a publishable article. It produced something more useful: a precise list of what was wrong with my design. This post covers the knowledge the agents work from, what happened in the run, what broke, and what I changed afterwards.

## Skills: reusing ideas, not files

Before writing any agent, I had to decide what the agents know. [InvestPal](/investpal/), my previous project, already had a library of analytical skills. I had built them by distilling investing books into checklists: Graham and Dodd and Bruce Greenwald on valuation and moats, Warren Buffett's letters on owner earnings, Howard Marks on second-level thinking and market cycles, Ray Dalio on debt cycles. Each skill ends in a required verdict and a set of style rules, so an agent cannot finish with a shrug.

I wanted the ideas, not a copy. InvestPal analyses single stocks for one user. Owl publishes a general house view on ETF exposures for beginner and intermediate investors. So the skills were adapted, not ported:

| Group | Skills | What changed |
|---|---|---|
| Thesis and macro | Second-level thinking, investment themes, market sentiment, debt cycle, macro regime, currency risk | Rewritten around a house view on an exposure. "What should the client do" became "what does this mean for a long-term ETF investor", which is often: nothing |
| Company analysis | Balance sheet, cash flow, earnings quality, valuation, moat, intrinsic value and four more | Method unchanged. Used to look inside an ETF's largest holdings, and as evidence in an article |
| New | ETF look-through, house rules, house voice, templates for briefs, theses and articles, fact-check procedure | Written for Owl |

Two adaptations are worth showing. The investment-theme skill forces every theme into one sentence before anything else happens: *X will happen by Y, and I will know it is happening because Z.* That sentence became the shape of every house thesis. And the second-level thinking skill, which asks what the price already assumes and why the house should be more right than the market, became the Risk Officer's whole job.

One more piece of plumbing mattered. Each agent can write only its own files. A hook checks every write against a list of allowed paths and blocks anything else. I tested it live before the run: the Editor could write its edit notes and could not write anywhere else.

## The run

The test week was 9 to 13 November 2026. The agents used only web search, on purpose: I wanted to see which data they reach for before building a data layer.

| Step | Agent | What happened |
|---|---|---|
| Research | Market Analyst, Macro Economist, in parallel | Two briefs, 43 and 44 source tags, 13 and 16 logged data gaps. About 8 minutes |
| Themes | CIO | Three candidates. Recommended: medium-term US Treasuries will beat cash over 12 months |
| **Gate 1** | Me | Picked the recommended theme |
| Funds | ETF Strategist | One US and one UCITS fund for the thesis and for the cash benchmark |
| Draft | CIO | Draft plus a registry entry. Its own check: "passes, narrowly" |
| Red team | Risk Officer | Round 1 `revise` with 8 required changes; round 2 `approve` with 5 fixes |
| Edit | Editor | Two passes; too long both times |
| Fact-check | Fact-Checker | Round 1 `fail` (43 verified, 1 corrected, 3 unverifiable); round 2 `fail` (49, 1, 1) |
| **Close** | Me | Stopped at the review limit. Nothing published |

Totals: 18 agent runs, about an hour of wall-clock time, about 1,5 million tokens across the subagents. My part was three decisions, a few minutes in total.

### The red team rebuilt the thesis

The most valuable thing the run did was change its own mind.

The CIO's first draft said medium-term Treasuries would beat cash because a 5% yield already priced the rate scare. It claimed yields could rise about 15 basis points (bps) before the bonds fell behind cash.

The Risk Officer took that apart. Cash rates are expected to rise too, it argued, so the bond fund's real cushion was closer to 6 bps than 15. The forecasters the draft called "the consensus" sat roughly where the house did, so there was no edge against the price. And the headline read like a buy recommendation. Its verdict was `revise`, with eight specific changes and a warning that round 2 would be `reject` unless the thesis was restated honestly.

The CIO accepted all eight. The thesis that came back made a much smaller claim: a patient holder is paid a small premium to hold medium-term bonds instead of cash. After fees, the expected lead was about 0,55 points, with a breakeven of about an 8 bps rise in yields and odds close to even. The headline changed from implying "buy" to "why we would not swap bonds for cash". That is a worse story and a far more honest one, and it is exactly what the review loop is for.

### The fact-checker refused what it could not trace

The Fact-Checker verified 49 claims against sources it fetched again itself. It also found a figure no source could support: an outflow of $836 million from a Treasury fund that the Market Analyst had picked up from a page that no longer existed. The figure was withdrawn, not kept. It also found the official name of an index written wrong, and a deficit figure cited to a think tank instead of the US Treasury.

## What broke

A dry run is meant to fail in useful ways. These were the useful ones.

**Problems were caught late.** The thin edge was visible at the theme stage, but the red team only saw it after a full draft. A question about how the thesis was framed came up at the very last check, after four drafting passes. Each late catch cost a loop.

**The workflow had gaps.** The Risk Officer's second verdict was "approve, but fix these five things", and the workflow had no such path. I improvised one. The agents that fixed their own briefs had no edit tool, so they rewrote whole files to change one row. Files weren't committed between steps, so I couldn't diff those rewrites.

**The Editor could not count.** It has no shell, so it estimated length by hand and was off by about 150 words. Two editing passes later the article was still over the target.

**A "verified" number was wrong.** While building tooling after the run, I fetched the fund prices the thesis started from straight from the issuer. Both differed from the values in the registry in the fourth significant digit, and the Fact-Checker had marked those values verified. Either a model misread the page or the issuer restated the values. Precise figures now get a digit-by-digit check.

**I was too strict.** I had made compliance a gate: house rules about what a thesis may name, a stop for any regulatory question. The run spent a full fix round restating the thesis to satisfy a question nobody could answer. That was the wrong trade-off for this stage.

## What I changed

| Change | Why |
|---|---|
| The Risk Officer screens the three themes before I pick | Move the edge debate before drafting, where it is cheap |
| New verdict: `approve-with-fixes` | Fixes go to the CIO without another red-team round; the Fact-Checker verifies them |
| The Editor flags only substance to the CIO | Length and wording are the Editor's job; they caused an extra loop |
| Fact-check round 2 checks only what changed | Half the work, same safety |
| Research agents can edit in place, and every step is committed | Every change becomes a reviewable diff |
| Briefs capped at about 2.000 words; articles 1.000 to 1.500 | The first brief was 4.600 words and every later agent read it |
| Compliance questions are notes, never a gate | Product quality first at this stage; clear "not investment advice" disclosures |
| A bond-fund branch for the ETF look-through | The equity version flagged a government bond fund as dangerously concentrated |

Two decisions went the other way. No paid data: index data from providers like ICE is licensed and expensive, so the agents keep using web search until Owl has its own data tools. And no mechanical scoring. I built a script that would mark each thesis right or wrong from fund prices at the horizon, then removed it the same day. A thesis can be right or wrong for many reasons, and reducing it to one number from one script is too deterministic. Every thesis still states what would prove it wrong; how outcomes get judged on the public track record is now an open question.

## What the data layer needs

The point of running on web search was to collect gaps, and the run logged about 40. Ranked by how often they came up, the data layer needs:

1. Same-day government bond yields, with last week's and last year-end's values for comparison.
2. What markets expect central banks to do, updated daily.
3. ETF data from the issuers themselves: costs, holdings, yields. Third-party sites disagreed on fund sizes.
4. Daily prices for indices, sectors, currencies and commodities.
5. Fund flows by fund.
6. A licensing flag on every data series, so the agents know what they may quote. One credit-spread series the Macro Economist used is licensed for internal use only.

Not all of it is needed yet. I'll run a few more weeks first, so the data tools get built from gaps that keep coming back rather than from one week's list.

## Next

The next run is a real one: the following week, with the new workflow, aimed at publishing. It will show whether moving the checks earlier pays off in fewer loops. The [project page](/owl-of-wall-street/) lists the series as it grows.

---

*Owl of Wall Street's analysts are AI agents, not licensed financial advisers. The thesis described in this post comes from a dry run that was never published; it is an example of the process, not investment advice.*
