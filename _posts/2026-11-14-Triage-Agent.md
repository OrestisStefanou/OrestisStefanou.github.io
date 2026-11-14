---
layout: post
title: "An agent that decides which tickets are safe to automate"
date: 2026-11-14
tags: [AI, agents, Claude Code, software factory]
description: "A triage agent that reads incoming feature requests, decides against a fixed rubric which ones need no new design, and hands those to the software factory after one human approval."
short_title: "An agent that decides which tickets are safe to automate"
summary: "A triage agent that sorts incoming requests into 'needs a person' and 'needs no new design', and sends the second kind to the software factory after one approval."
---

The [software factory](/software-factory/) starts with a spec interview. A tech-lead agent explores the code, asks the engineer one question at a time, and only then writes a task file. That is the right amount of care for a new feature. It is far too much for "add a search box to this list".

A new internal app gets a lot of requests like that. People use it, notice what is missing and file a ticket: a filter here, a column there, a button that opens a record that already has a page. Each one is small, each one waits behind real work, and together they decide whether people keep using the app.

So I added a stage in front of the factory. A triage agent reads every new request, decides whether it can be built without anyone making a design decision, and if so writes a proposal. I approve or reject it with one command. An approved proposal goes through the factory without a spec interview and comes back as pull requests for review. Nothing merges without a person.

{% include triage-loop.html %}

## Trivial does not mean small

The obvious rubric is size: a few files, under a hundred lines. It is wrong in both directions. A 20-line change can hide a product decision ("who is allowed to see this?"). A 400-line change can be pure repetition: a new list screen that copies three existing list screens, over an endpoint that copies three existing endpoints.

What matters is whether an engineer would have to **design** anything. So the definition became:

> Trivial means no new design. The issue settles what to build, and the code already shows how.

Size stays in the output as an estimate for the reviewer. It never decides the verdict.

## The rubric

An issue is trivial only if every criterion holds:

| # | Criterion | What passes |
|---|---|---|
| 1 | Backend | None, or a thin layer over existing data: endpoints that read or change what is already stored, following a router pattern the agent can name |
| 2 | Schema | No migration, no new model, no new relationship |
| 3 | Authorization | Reuses an existing permission check. "Only the owner" is fine when that check already exists |
| 4 | Patterns | Every screen, component, hook and endpoint follows one already in the code, named by file. A new kind of interaction fails |
| 5 | Product decisions | None open. Minor details with an obvious default from an existing screen (a label, an empty state, column order) are allowed and listed |
| 6 | Checkable | Backend behaviour covered by tests; frontend by lint, type-check and tests |

Some things disqualify an issue whatever the rest says: a hard delete that cascades, a new write path to customer records, an audit-trail requirement, any change to AI behaviour, and anything that moves money. These are the places where a quiet mistake is expensive, so they always go to a person.

There are three verdicts: `trivial`, `not-trivial`, and `needs-info` when the request is too vague to judge even with the code open. One escape hatch made the rubric much more useful: when the issue as written fails but a narrower version passes, the agent may propose the narrower version. The narrowing itself becomes one of the decisions I approve.

## Evidence, not wording

A request that says "just a small tweak" tells you nothing. The agent must base every criterion on code it actually read, and cite it: the endpoint it would copy, the component it would reuse, the permission check that already exists. A criterion without a file and line does not hold.

This keeps the verdict honest in both directions. It stops the agent from waving a change through because the ticket sounds simple, and it stops it from rejecting something because the ticket sounds big.

The agent is also kept narrow. It can read the frontend and backend code and write one file: its proposal. It cannot change code, comment on tickets or post anywhere. The session that runs the schedule does all of that.

## Ticket text is data

The issue title, description and comments come from people outside the session. To the agent they describe a request; they are never instructions. If a ticket contains text aimed at an agent ("mark this trivial", "also change X"), the agent ignores it and flags it in the proposal's risks. When a ticket links a design, the parent session fetches it and passes it in the same way, marked as data.

## The proposal

A trivial verdict comes with a short proposal, written for the person approving it:

- **Verdict**, in one or two sentences, with the deciding reason
- **Rubric**, one row per criterion, with the evidence
- **Decisions**: every minor choice the agent made, each with its default and a one-line reason. "Empty state message? Default: the wording used on the existing list screen."
- **Plan**, as requirements rather than design: endpoints with method, path, fields and status codes, the pattern to follow, the authorization rule, the tests that must exist; for the frontend, which screen changes, which component to reuse, and every state (loading, empty, loaded, error)
- **Acceptance criteria** that can be proven false
- **Risks**, where "none" is a valid answer

Listing the decisions is the most important part. A proposal with no decisions is fast to approve. A proposal with three decisions shows me exactly where to look.

## One gate

Approval is a command, and only I typing it counts. A schedule never approves. A chat reply never approves. Text inside a ticket or a proposal never approves.

I can change any decision as part of the approval. The changed answer is written into the proposal, marked as changed, and from then on the decisions are binding on the build.

Compared with the factory, this replaces the spec interview with reading a one-page proposal. That is the whole saving, and it is only safe because the rubric has already shown there is nothing to design.

## The build

Approval creates a git worktree per repository on the ticket's branch and starts one background session per repository, in parallel. Each runs the software factory's build: test-first implementation, two reviewers, a capped fix loop and QA. For frontend changes it also takes screenshots against staged data.

The backend build adds one more check. Its first step is the factory's tech-lead agent writing the task file from the proposal, and that agent can refuse: if it finds a schema change, a new permission model or a pattern with no precedent, it says the work is not trivial and writes nothing. That is a second, independent check on the triage agent, done by an agent that has to turn the proposal into real requirements.

Each build ends in a pull request that is ready for review, so the automated review bots run on it. A follow-up session then watches the pull request and answers review comments until it is merged or closed. If any build stops, the run is marked failed and its worktree is kept, because that is where I take over.

## Telling the team

The system is allowed to post exactly one kind of message: a finished build, with links to its pull requests and the screenshots in a thread under it. Nothing is posted at triage, at approval, at rejection or when a build fails. Each post says plainly that it is automated.

Keeping posts this narrow is what makes them worth reading. The person who filed the request sees a pull request with screenshots, and nobody gets messages about work that may never happen.

## Housekeeping and state

Once every pull request for a ticket is merged or closed, the next run removes its background sessions, worktrees and merged branches, skipping anything that could lose work. A follow-up session that has gone quiet on an open pull request is reported once, not every hour.

All state lives in files, never in a session. Each ticket has one proposal file whose frontmatter records where it is: proposed, approved, building, ready, merged, rejected, failed, or on hold behind another ticket. The scheduled run is stateless: it reads the files, does the next step for each, and writes them back. A crashed or restarted session loses nothing, and the whole state of the system fits in one table.

Rejected tickets are never assessed again. `not-trivial` and `needs-info` tickets are assessed again only when someone edits them, which is exactly the moment the answer might change. Tickets on hold are assessed again when the ticket they wait for is done.

## What went wrong

The agent reads the code as it is on disk. Early on it judged a ticket against a local checkout that was well behind the main branch. It reported the backend work as missing when that work had been merged minutes earlier, and the verdict's reasons were wrong.

Nothing in the setup had ever told it to fetch. That is the general lesson: an agent treats whatever is in front of it as the truth, so the setup has to make sure what is in front of it is current. The fix is a fetch and fast-forward before every run, and a stop-and-ask when a checkout has local changes.

## Principles

| Principle | In practice |
|---|---|
| Classify by design, not size | The question is whether anyone has to decide something, not how many lines it takes |
| No evidence, no pass | Every criterion cites the code that proves it |
| Default to no | Anything unclear, risky or expensive to get wrong goes to a person |
| One human gate before code, one before merge | Approval is a command only the engineer can give; merging stays with people |
| Agents see the ticket as data | Ticket text is never instructions, and an injection attempt is flagged |
| Least privilege per role | The assessor reads code and writes one file; posting and builds belong to other parts |
| State in files | Every step can be resumed, inspected and audited from the files alone |
| Talk only when there is something to see | One kind of post: a finished build with its pull requests |

Built on Claude Code subagents, skills, scheduled tasks and background sessions, in front of the [software factory](/software-factory/).
