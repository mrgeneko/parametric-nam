# Proposal: deploying and routing work across multiple machines

> **STATUS: PROPOSAL. None of this is implemented.** Everything described under
> "Recommendation" is a plan, not a feature. What exists today is `distribute_pull.py`
> ([scripts.md](scripts.md#distribute_pullpy--hand-rendering-chunks-out-as-workers-free-up)):
> one control machine holding SSH access to every worker, handing out chunks as they free up.

## What prompted it

Rendering a 648-combination full-amp dataset across four machines took 13 hours and worked,
but the setup around it did not scale to a person's patience. The complaint that started this
was not reliability — SSH never failed during the run — but that **adding a machine to the
fleet is a multi-step manual chore**, repeated per machine.

Both halves of that are worth separating, because they have different answers.

## What actually went wrong during a real run

| symptom | root cause | architectural? |
|---|---|---|
| one worker produced 0 of its combinations for 7.5 h | no per-combination visibility; the controller only judged completed chunks | **yes** — fixed since |
| first launch failed in under a second | the device was described twice: a config for single-machine, hand-typed flags for sharded | **yes** — fixed since |
| absolute paths from the config were unusable | heterogeneous home directories across the fleet | **yes** |
| a worker's simulator appeared to be missing | non-standard install layout, plus an env var it needed | no — inventory |
| a worker was 472 commits stale | nothing verified code version at dispatch | **yes** |
| a metadata column came out unusable | the render began before a fix landed; invisible in the artifacts | **yes** |
| 19 GB of results were transferred twice | nothing modelled *where a dataset should end up* | **yes** |

The first two have been fixed (per-combination pacing; `distribute_pull --config`). The rest
motivate what follows.

## Counting the real onboarding cost

The premise deserves testing, because SSH turns out to be the *cheapest* of the steps:

| # | step to add a machine | can it be eliminated? |
|---|---|---|
| 1 | install the VPN/mesh network | no — but you do this anyway |
| 2 | SSH key trust (keys, `known_hosts`, host aliases) | **yes, one command** — see below |
| 3 | clone the repo | `setup.sh` |
| 4 | build the Python venv | `setup.sh` |
| 5 | **build the .NET oracle (`livespice-cli`)** | **the expensive one** |
| 6 | register the worker with the scheduler | inventory file |

Steps 3–5 dominate, and no choice of transport touches them. Any proposal that replaces SSH
but leaves the toolchain build in place has addressed the smallest line item.

## Recommendation

### 1. Mesh SSH — removes step 2 for one command per host

If the fleet is already on a mesh VPN (this one uses Tailscale), enabling its SSH mode makes
authentication an identity/ACL question rather than a key-distribution one: no
`authorized_keys`, no `known_hosts`, no per-host aliases. Its DNS also gives every node the
same names, which removes a real failure: a worker that cannot resolve the *controller's*
private aliases cannot be told to fetch from a peer.

No code change. Do this first regardless of what else is adopted.

### 2. An inventory file — removes step 6, and fixes a scheduling bug

Per-host facts currently live in command-line flags and operator memory. Every one of these
was rediscovered by hand during the run: which checkout a host uses, which env vars its
simulator needs, how many **physical** cores it has, which backends it can run at all, and
whether it is fast enough to finish a single combination inside the timeout.

```toml
[hosts.linux-1]
address      = "linux-1"                    # mesh DNS name — resolves the same on every node
repo         = "~/work/parametric-nam"
cores        = 6                            # PHYSICAL, not logical
backends     = ["livespice", "ngspice"]

[hosts.linux-2]
address      = "linux-2"
repo         = "~/work/render/parametric-nam"   # a non-standard layout, recorded once
cores        = 6
env          = { DOTNET_ROOT = "~/.dotnet" }
max_render_s = 2400                         # fine for pedals; cannot finish a full amp

[hosts.mac-1]
address      = "mac-1"
repo         = "~/work/parametric-nam"
cores        = 8
backends     = ["livespice", "ngspice", "ltspice"]   # ltspice is macOS-only
```

**`cores` is physical on purpose.** A host advertising 12 logical cores from 6 SMT pairs was
given 10 concurrent renders; each got well under a core, every combination fell just short of
its timeout, and the machine produced nothing for 7.5 hours while looking busy.

**Capability is declarable rather than discovered the hard way.** `max_render_s` states "this
box is useful for pedals, not for a full amp at oversample 8" — which is a true and useful
thing to say about a machine, and is not the same as "broken".

This is orthogonal to the per-device `<stem>.backends.toml` sidecars: those record which
backend a **circuit** needs; this records which backends a **host** can run. A job needs the
intersection of the two.

**It should be generated, not hand-written.** Every field above is measurable — core counts
from `sysctl`/`lscpu`, the checkout by looking for it, the env by testing whether the
simulator runs, the backends by probing for each, `max_render_s` by timing one short render.
A `--probe-hosts` mode should emit a reviewable file the way `scaffold_config.py` emits a
device config: measured, annotated, and yours to correct. Requiring a hand-written file with
no discoverability is a mistake this project has made before.

**Where it lives.** The loader, the flag, the docs and an `examples/fleet.example.toml` are
public and part of this repo. *Your* inventory is per-machine state, not project content: it
defaults to `~/.config/parametric-nam/` (mirroring `~/.cache/parametric-nam/`) and
`--inventory` overrides it. It must stay **optional** — explicit `--worker` flags keep
working, so a first run never depends on a file the user does not know to write.

**One thing it must not become.** Recorded throughput belongs in the inventory for
*estimation and reporting*, never as a scheduler input. `distribute_pull` is pull-based
precisely because static weighting failed: its own history records a 1.6 h job stretched to
8.9 h that way, and measured rates from the recent run tracked neither core count nor RAM.
The schedule stays the measurement.

### 3. Containers — only for the Linux workers, and only for step 5

Containerising is the obvious answer to "every machine needs an identical toolchain", and it
is the right one **for Linux hosts**: a prebuilt image collapses steps 3–5 into a pull.

It is the wrong answer for macOS hosts, for two concrete reasons:

- Docker on macOS runs a **Linux VM**. For CPU-bound circuit simulation that is a real
  throughput loss, and it would land on the fastest machines in a typical Apple-silicon fleet.
- The `ltspice-deck` backend is a macOS-native application. It cannot be containerised at all.

So the honest recommendation is a **split**: native on macOS, image on Linux. That is less
tidy than "containerise everything" and is what the measurements support.

**What containers are often reached for and should not be**: guaranteeing that every worker
runs the same code. That is cheaply solved by having the scheduler verify a commit SHA and a
simulator version *at dispatch* and refuse mismatched workers. That check would have caught
both the 472-commit-stale checkout and the render that silently began before a fix landed —
natively, on every platform, for a fraction of the effort.

### 4. Pull-based agents, a durable queue, and a dashboard

Invert the flow: instead of a controller pushing work over SSH, workers poll a coordinator,
lease a chunk, heartbeat per combination, and report results. At this scale the coordinator
can be a small HTTP service over SQLite.

Justified by three things — **note that onboarding is not one of them**, once mesh SSH is in
place:

- **The controller is a single point of failure.** A 13-hour render currently lives or dies
  with one machine staying awake.
- **Observability.** Queue depth, per-worker combinations/hour, live progress and failure
  reasons — the information that had to be reconstructed by hand while a worker sat idle for
  7.5 hours. This is where a **web interface** naturally belongs; it falls out of having a
  queue and a database rather than being a separate project.
- **Join and leave.** A machine that boots takes work without editing a command line.

Keep SSH for bootstrap — starting the agent and shipping the repo. It works, and the mesh VPN
already provides the network.

### 5. Data gravity

The queue should carry a **destination**. Results should land where training will run, once.
In the recent run a 19 GB dataset was collected onto the controller and then transferred again
to the machine that would train on it, because nothing in the system expressed where a dataset
belongs. Options: an S3-compatible store on the training host, or simply a per-job "sink host"
that workers write to directly.

## What not to do

- **Kubernetes, Nomad, Celery** — operational burden out of proportion to a handful of
  machines and one operator.
- **Ray** — attractive on paper (Python-native, heterogeneous, has a dashboard), but it
  requires matching Python *minor* versions across nodes; a fleet assembled over time will not
  have that, and pinning it fleet-wide is its own chore. It is also heavy for work that is
  ultimately a subprocess launching a simulator.
- **Containerising the macOS hosts** — see above.
- **Replacing SSH before enabling mesh SSH** — it is one command, and it removes most of the
  complaint without touching any code.

## Sequencing

1. Mesh SSH. One command per host, no code.
2. Inventory file with `--probe-hosts` generation. Fixes the physical-core bug and stops
   rediscovering per-host facts.
3. Dispatch-time version verification. Cheap; closes the stale-worker and version-skew holes.
4. Pull agents + queue + dashboard. The largest piece; justified by observability and removing
   the single point of failure.
5. Destination-aware results.
6. A Linux worker image, only if Linux machines are added often enough to pay for it.
