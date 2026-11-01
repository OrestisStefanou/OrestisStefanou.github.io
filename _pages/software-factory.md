---
layout: page
title: A self-improving software factory
eyebrow: Case study
permalink: /software-factory/
description: "How I run coding agents from a rough brief to a monitored production change, with an engineer deciding at exactly two points, and a weekly measurement that makes the whole loop better over time."
---

Coding agents can write code faster than anyone can review it. Speed stopped being the hard part a while ago. The hard parts are knowing what to build, knowing that what was built is right, and knowing that it still works once it meets production.

This factory is how I deal with those three. It is a set of agents, skills and workflows on top of Claude Code that takes a ticket from a rough brief to a monitored production change. The engineer makes two decisions along the way: **what to build**, and **whether it ships**. Everything between those two gates runs unattended, and everything after the second one is watched.

{% include factory-loop.html %}

## The loop

| Stage | Who acts | What comes out |
|---|---|---|
| Spec | Engineer and a tech-lead agent | An approved task file: rules, edge cases, acceptance criteria, production signals |
| Build | A workflow of agents, unattended | A pushed branch that passed review and QA, plus a review guide |
| Review and merge | Engineer, CI and review bots | A merged pull request |
| Deploy and watch | A monitoring agent, on a schedule | Checks against production, and an alert only when one newly fails |

Two more pieces sit outside the per-ticket loop: a **memory** that every agent reads and writes, and a **weekly measurement** that scores the whole loop and decides what to change next.

## Gate 1: the spec

Most bad code starts as an unclear spec, so the factory spends its effort there first.

A tech-lead agent explores the codebase before it asks anything. Where the size of a change matters, it also reads production read-only, as counts and distributions only, never personal data. If a "this can't happen" state already exists in production, it becomes a requirement instead of a surprise.

Then it interviews the engineer one question at a time, each with a recommended answer, until every branch of the design is settled. The output is a task file with:

- business rules and the invariants that must hold
- edge cases, including the ones production data revealed
- acceptance criteria that map one to one onto tests and QA scenarios
- **production signals**: what healthy looks like after deploy
- an authorisation section whenever access rules change: who can do what, and what else a new permission unlocks

The agent also checks whether the work is really one task, both while clarifying and again on the finished spec, because specs grow while they are being written.

The engineer approves the file. That is gate 1.

## Build: unattended, with caps

One command starts a workflow that drives the approved task to a pushed branch. It never merges and never opens a pull request.

**Spec review comes first.** A fresh agent reads the task file cold, against the repository's standards, schema and code. It blocks only on defects it can prove: rules that conflict, a rule that contradicts a written standard, a rule that cannot run on the real schema, a criterion nothing could assert, or a task that should ship as two. Everything else becomes a warning for the implementer. This step exists because the most expensive failures used to be spec defects discovered after a full implementation run.

**Implementation is test-first.** A failing test, proof that it fails for the right reason, then green.

**Review runs two reviewers in parallel.** One checks the spec: every acceptance criterion has code and a test, the invariants hold, and the team's blocking rules are followed. The other is a general bug and security review that catches what no checklist names. Findings are labelled; only the important ones block.

**The fix loop is capped at three rounds.** After the first round, reviewers read only the commits since the last review, so the loop converges instead of re-finding old code.

**QA runs on real services, capped at two rounds.** One scenario per acceptance criterion, and a scenario passes only when every side effect is observed. Failures are classified as code problems, which go back to the implementer, or spec problems, which go back to the engineer.

If anything goes wrong that the agents should not decide alone, such as a provable spec defect, a hit cap or an agent that died, the run stops and hands back with a report that names the stage, the reason and the question to answer. It never guesses.

## Gate 2: review with a guide

When the run finishes, a fresh agent that took no part in it reads every report and the diff, and writes a review guide for the engineer:

- a risk tier
- **what the checks would miss**, which is the section to read first
- a reading order for the diff, starting with the files that matter
- evidence for each acceptance criterion
- where the code deviated from the spec, and why
- leftovers: nits and pre-existing issues that were deliberately not fixed

The engineer reviews the diff with the guide open, opens the pull request, deals with CI and the review bots, and merges. That is gate 2.

## Deploy and watch

After deploy, the production signals from the spec become checks: queries that return nothing when the system is healthy. Each check is dry-run once, then runs on a schedule for about ten days. The engineer is notified only when a check newly fails, not every time it runs. When a watch expires, the one or two checks worth keeping are drafted as permanent alerts.

The monitoring agent reads production read-only. Every tool call it makes passes an allowlist, every query has a hard timeout, and its reports carry counts and internal ids, never customer data.

## Memory

Agents forget everything between sessions. The factory uses two kinds of memory to fix that.

**Work-scoped memory** lives in [Loci](https://github.com/OrestisStefanou/Loci), filed by project and task. The spec agent saves the decisions it settled. The implementer saves deviations and why. Reviewers save findings that were settled between rounds, so they are not raised again. The monitoring agent saves check history. Anyone, human or agent, who picks up the ticket later starts from what was already decided.

**Always-apply rules** live in plain markdown that each agent loads when it starts: conventions, gotchas, things to never do again.

Both stores drift, so a curator agent audits them now and then. It finds memories that are stale, duplicated, misplaced or missing, proposes each change with evidence, and applies only what the engineer approves, with backups.

## What makes it self-improving

A factory that only runs gets no better. Once a week, a measurement pass reads the run records, the branch history, the pull request comments and the watch logs, and scores the loop:

- every change requested after gate 2, why it was needed, and which stage should have caught it
- every run that handed back early, and why
- cost per task
- bugs that reached production within 30 days of a merge

Agents propose the labels, and the engineer confirms them. The output is not a dashboard for its own sake. Each recurring miss turns into a change at the stage that should have caught it: a new rule for a reviewer, a new defect class for spec review, a new QA scenario type, a new memory. The spec review step itself came out of this process.

## Principles

| Principle | In practice |
|---|---|
| Humans decide, agents do | Two gates. Everything else is delegated, and nothing merges without a person |
| Every loop has a cap | Three review rounds, two QA rounds, then hand back instead of guessing |
| Fresh eyes | Spec review and the review guide are written by agents that took no part in the work |
| Block only on proof | A reviewer that blocks must show the defect. Opinions become warnings |
| Production is the truth, read-only | Specs, QA and monitoring look at real data shapes, with no write access and no personal data |
| Measure before adding agents | New stages are added because the weekly numbers show a gap, not because they sound useful |

Built on Claude Code subagents, skills, hooks and workflows, with Loci for memory.
