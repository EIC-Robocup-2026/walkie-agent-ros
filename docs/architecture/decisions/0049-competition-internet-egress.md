# ADR-0049: Limit Competition internet access to named sources and web-search snippets

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

[ADR-0017](/architecture/decisions/0017-rmw-zenoh-domain-23.md) runs the
agent on the robot PC and states that only the LLM calls leave it. At
Competition, the self-evolving agent needs more than that:

- `look_up_online` answers long-tail questions from web-search snippets.
- Pre-baked and generated tools need live data, such as weather, exchange
  rates and Wikipedia summaries.
- `fetch_sound` finds short sound clips on Wikimedia Commons.

Everything that comes back is untrusted. An edited Wikipedia extract can
carry instructions (indirect injection, T2). A query can carry what the
robot saw or heard (exfiltration, T4). A snippet can end up spoken aloud
(unsafe speech, T5). A crowd can drain a quota (resource abuse, T6). The
threat model lives in [Access Levels](/architecture/access-levels.md).

RuleBook 2026.2 counts cloud services reached over the Arena Network as
external computing, to be declared to the Technical Committee one month
ahead. It does not guarantee internet, and the robot must stay safe
without it.

**Decision**

At Competition, the parent process makes every outbound request. Generated
code never opens a socket. Each request goes through one broker to one of
these:

- **OpenRouter**, for the LLM calls, as today.
- **Named sources with pinned templates.** A template fixes the host, the
  path, the method and the allowed query parameters. Only GET over HTTPS
  on port 443. The six data sources are:
    - `open_meteo`: `api.open-meteo.com/v1/forecast`
    - `open_meteo_geocode`: `geocoding-api.open-meteo.com/v1/search`
    - `frankfurter`: `api.frankfurter.dev/v1/`, plus `latest` or a date
    - `wikipedia_summary`: `en.wikipedia.org/api/rest_v1/page/summary/`,
      plus exactly one path segment
    - `commons_search`: `commons.wikimedia.org/w/api.php`, with
      `action=query` and `prop=imageinfo` only
    - `commons_media`: `upload.wikimedia.org/`, returned as a handle
- **`web_snippets`,** a seventh named source, with one template per search
  provider. Each template fixes the result count, strict safe search and
  the language, and allows no extra snippets, page content or images.
  Redirects are refused. The parent adds the provider's key after the
  template check.
    - **Brave Search, first:** a GET to
      `api.search.brave.com/res/v1/web/search` with `q`, and with `count`,
      `safesearch=strict` and `search_lang` fixed. `extra_snippets` stays
      off.
    - **Tavily, as the fallback:** a POST to `api.tavily.com/search`. The
      JSON body is fixed except for `query` and `language`:
      `search_depth = "basic"`, `max_results = 3`, `chunks_per_source = 1`,
      `safe_search = true`, and no answer, raw content or images. This is
      the only POST at Competition, and only the parent's search client
      makes it. Generated code still has GET alone.

No moderation service is reached. The speech gate is local only
([Competition Mode](/architecture/competition.md#speech-gate)).

[Competition Mode](/architecture/competition.md#named-sources) lists each
template's parameters. `web_snippets` gets the named-source caps plus its
own:

| Cap at Competition | All named sources              | `web_snippets`, in addition              |
| ------------------ | ------------------------------ | ---------------------------------------- |
| Per invocation     | 5 requests                     | 2 searches per turn or invocation        |
| Per minute         | 30                             | 6                                        |
| Per hour           | not capped separately          | 120, shared by all visitors              |
| Concurrency        | 1 per host                     | 1 per host, as for all named sources     |
| Response           | 1 MiB decoded                  | 3 results, 1,500 characters after shaping |
| Timeouts           | connect 3 s, total 8 s         | connect 2 s, total 3 s per provider; 4 s per search |

The broker checks every request:

- It normalizes the URL and matches it against the source's template.
- It resolves the host once, refuses local and on-link addresses, and pins
  the vetted IP, so the name is never re-resolved.
- It follows at most 3 redirects, only within the same source, and none
  for `web_snippets`.
- It caps decoded bytes, so a gzip bomb stops at the cap.
- It sends a fixed User-Agent with a team contact.

**Search providers.** Brave comes first because it has its own index,
strict safe search, a GET with one header, and about 0.6 s per search.
Tavily takes about 1.7 to 1.9 s at basic depth, and its safe search is off
unless the template sets it. Tavily is kept as the fallback so that one
provider's outage does not switch web search off for an event.

- One attempt per provider, each with connect 2 s and total 3 s. The
  whole search stops at 4 s.
- Tavily is tried only after a fast Brave failure: a DNS, connection or
  TLS error, or an HTTP 401, 403, 429 or 5xx status. A Brave timeout is
  not retried at Tavily, so a fallback never doubles the wait.
- A failed provider is skipped for 60 s, or for the session on 401 or
  403, so the next searches go straight to the other one.
- Keys are per provider and per role: `WALKIE_BRAVE_KEY_ROBOT`,
  `WALKIE_TAVILY_KEY_ROBOT`, `WALKIE_BRAVE_KEY_SIM` and
  `WALKIE_TAVILY_KEY_SIM`. Only the parent reads them. Separate keys keep
  sim runs off the robot's quota. A provider whose key is unset is skipped,
  so a sim machine can run on Brave alone.

Prices were checked on the providers' pages on 2026-10-22. The latencies
are third-party benchmark means measured in the US, not at the venue. The
cloud LLM (the Competition coder included) and both search providers are
declared to the Technical Committee one month ahead.

Rejected:

- **Named sources only.** Every long-tail question would need a new source
  and a generated tool, at 20 to 35 s each, instead of one search answered
  in an estimated 7 to 8 s end to end.
- **GET to any public host.** Every page becomes an injection channel and
  a source of misinformation, and every query string an exfiltration
  channel. No template can bound either. It would also give the coder open
  web access again.
- **Brave alone.** One secret and one service to declare, but a Brave
  outage would turn `look_up_online` off for the whole event.
- **Tavily alone.** No card and 1,000 free credits a month, but about three
  times Brave's latency, safe search off by default, and a POST for every
  search.
- **A cloud moderation check in the speech gate.** It would have read
  echoed, generated and tainted text before it was spoken, and switched
  creation and `look_up_online` off whenever it was down. OpenRouter has no
  moderation endpoint, so it would have been a guard model call, such as
  `meta-llama/llama-guard-4-12b`, which one provider serves. It is dropped,
  and the gate keeps only its local checks. So text built from clean words
  (an insult, a defamatory line, or bad advice in a snippet) can reach the
  speaker. What limits it instead is listed in
  [Competition Mode](/architecture/competition.md#speech-gate).

**Consequences**

- Amends ADR-0017: the LLM calls are no longer the only traffic that leaves
  the robot PC. It also holds the robot's search key, and its outbound
  traffic is limited to the hosts above.
- Offline, composing, creating, `fetch_sound` and `look_up_online` fail
  closed. Reusing existing tools keeps working. With the network up, only
  the kill switch, the caps, or both search providers failing turn
  `look_up_online` off.
- Each search costs one provider request, or two when it falls back. 120
  per hour bounds an
  8-hour booth day at 960 searches, or about $5 at Brave's listed $5 per
  1,000. Brave's $5 of free credit a month covers about one such day.
  Tavily's 1,000 free credits a month cover fallback traffic.
- Two secrets on the robot, two services to declare and two sets of terms.
  Brave allows storing results only on a plan with storage rights, so
  traces must mask snippet text. Tavily's terms on storage and LLM use are
  checked before phase 6.
- One visitor can use up the hour's searches. Per-visitor caps would need
  an identity the robot does not have.
- Adding a source is a reviewed code change, never a runtime decision.
  Only people edit the list. The coder cannot add to it.
- Snippets are never stored. Traces keep the query's hash and length, its
  language, the result count and the site names.
- Not yet tested in the prototype: path normalization and the templates,
  the local and on-link address checks, live TLS over the pinned IP, and
  the IPv6 fallback to the next vetted address.
- How the Technical Committee counts cloud APIs against its limit of five
  external computing resources is unknown. Recheck all of this against the
  2027 rulebook.

**See also:** [Competition Mode](/architecture/competition.md),
[Access Levels](/architecture/access-levels.md),
[ADR-0047](/architecture/decisions/0047-sandbox-and-bridges.md)
<!-- --8<-- [end:body] -->
