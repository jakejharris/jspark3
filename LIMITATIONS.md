# Limitations

The v2.0.2 candidate adds only the image checkpoint and GIF fixes. Live cache validation is pending; see
[the evidence](release/v2.0.2/ROOTCAUSE.md). Timing figures below were measured on v2.0.1, not this candidate.

## API

The server speaks the OpenAI chat completions API (`POST /v1/chat/completions`, `GET /v1/models`), with these known
differences. Check them against your client before relying on it.

1. **`stop` is ignored.** A reply ends at the model's end of turn or at `max_tokens`.
2. **`response_format` is ignored.** JSON mode and `json_schema` are not enforced, so the reply is free text.
3. **Identical prompts without a `seed` return identical outputs, even above temperature 0: a chat app's regenerate
   returns the same reply, and two users who send the same prompt get the same answer.** Send a different `seed` with
   each request when you want a different sample.
4. **`n`, `logprobs`, presence and frequency penalties and `logit_bias` are ignored.**
5. **A wrongly typed field, such as a string `temperature`, may return HTTP 500 instead of 400.**
6. **Non-streaming requests send nothing until the reply is complete.** Behind a proxy with an idle timeout, use
   `stream: true`.
7. **The `model` field is not validated; every request is served by GLM-5.3 Flash.**
8. **Without the draft model, a long conversation that includes images may not be saved to the disk session cache,
   and each saved state takes more memory, so fewer long conversations stay cached.** Returning to such a
   conversation after it has left the memory cache can take as long as its first prompt. Text-only conversations of
   about 40,000 tokens are saved; longer text-only conversations were not tested.
9. **Reasoning is always on, and no setting turns it fully off; v1.8.4 had it off by default.** A request that sets
   no reasoning effort runs at High, and the lowest effort is low; a top-level `reasoning_effort: "none"` and
   `chat_template_kwargs: {"enable_thinking": false}` are both treated as low, so a reply can still begin with a
   short reasoning passage. Even at low effort, a small `max_tokens` can be used up by reasoning and return no
   visible text; allow a few hundred tokens or more.
10. **With the draft model on, a short text request that arrives while no reply is streaming may wait up to 25 ms for
    a second request before its prompt is read.** Without the draft model, prompts that arrive together are not read
    together in one batch, and first tokens arrive later: with the base weights and 8 requests at once, the median
    first token arrives after 1.9 s, with visible text 0.41 s later (median over the replies that showed text),
    against 0.54 s with the draft model, where the median first visible text arrives at 1.5 s; in both sets, 3 of 24
    short replies spent their 96-token limit on reasoning and showed no text. Reading prompts that arrive together in
    one batch without the draft model is outside this release; no target version is assigned. With 16 requests at once, twice the server's 8 reply
    slots, total output with the draft model on is about 10 to 15% lower than with 8 (about 15% with the base
    weights, about 10% with the refusal-removed (ablit) weights), because the second eight prompts are read in small
    steps while the first eight replies stream. A streaming reply can occasionally pause between updates, and the
    pauses are longest while a long new prompt is being read: the longest measured pause was about 0.75 seconds, with
    a 36,180-token prompt. While a prompt of that length is being read, one step can pause every streaming reply at
    once for up to about 0.53 seconds.
11. **JSpark3 v2.0.2 saves the state at the end of each prompt it reads, whichever client sent it, and reuses it when
    a later prompt starts with that entire earlier prompt, such as the next turn of a conversation; it then reads
    only the rest.** Sharing only a system prompt is not enough: no state is saved where a system prompt ends, so a
    prompt with the same system prompt but a different first message is read in full. With the draft model on, it
    also skips reading a prompt that exactly repeats the latest prompt of a conversation, such as regenerating the
    latest reply, while that state is still in memory. With the draft model on, only the latest state of each
    conversation stays in memory, so regenerating or resending an earlier turn after later turns have been sent does
    not get this shortcut. Such a request resumes only from a shorter state that the disk session store has finished
    saving. The store saves in the background while the server is idle and may not yet hold a given turn, or may have
    skipped it; in testing, these regenerations read the whole prompt again. Without the draft model, an exact repeat
    is never skipped, but earlier turns' states can stay in memory until evicted, so regenerating a later turn can
    resume from the previous turn's state. Regenerating the first reply of a conversation after later turns, or
    resending it without the draft model, reads the whole prompt again, because saved state is reused only when it is
    shorter than the new prompt.
12. **Conversations that share only a system prompt do not share cached work.** Saved state is matched by prompt
    content, not by conversation: a prompt reuses an earlier prompt's state only when it starts with that entire
    earlier prompt, whichever conversation sent it. No state is saved at the end of a system prompt, so a new
    conversation that starts with the same system prompt as an earlier one, but has a different first message, reads
    its whole prompt again. That fix is outside this release; no target version is assigned.
13. **A client that disconnects while its connection's socket number is 1024 or higher is not detected, so its
    generation runs to completion and holds its slot.** Normal connection counts do not reach this; very many idle
    keep-alive clients could. That fix is outside this release; no target version is assigned.
14. **If `max_tokens` cuts off a tool call, `finish_reason` is `length` (or `tool_calls` if an earlier call in the
    same reply was complete), the cut-off call is left out of the final `tool_calls`, and its raw text is returned in
    `content`.** When streaming, its name and partial `arguments` (incomplete JSON) have already been sent. Raise
    `max_tokens` for tool use.
15. **The `usage` block in replies does not include `prompt_tokens_details.cached_tokens`.** The number of prompt
    tokens the server reused from saved state is reported in the reply's `tensorfold.cached` field instead (in the
    final chunk when streaming). For a request that forces a tool call, this count can be too high, even above the
    prompt's length. v1.8.4 returned this field, so a client that reads it must switch to `tensorfold.cached` when
    upgrading. Both fixes are outside this release; no target version is assigned.
16. **v2.0.2 does not support per-request cache isolation.** It ignores the `cache_salt` request field, and all
    clients of one server share its saved prompt state. A request whose prompt starts with another client's entire
    earlier prompt reuses that state, which shows in the reply's cached-token count and in a faster first token.
    v1.8.4's engine honored `cache_salt`, so a deployment that relied on it to keep clients apart is no longer
    isolated after upgrading. If clients must not learn about each other's prompts, give each one its own server with
    its own session folder. Per-request isolation is outside this release; no target version is assigned.

## Running it

- **No authentication, no CORS policy.** Rank 0 listens on loopback by default. Use an SSH tunnel or an
  authenticating proxy ([INSTALL.md](INSTALL.md)).
- **All three boxes must use the same weights, draft model and session tier.** The engine refuses to start when the
  draft model or session tier differs between boxes, and `scripts/wait-ready.sh` names the ranks. It cannot tell the
  base and ablit weights apart (they have the same shapes), so a box started with the other variant goes undetected
  by the scripts; `scripts/status.sh` on each box shows its variant.
- **The session tier keeps conversation state on disk** (up to 64 GiB per box) until it is evicted or you clear it.
  `SESSION_TIER=off` keeps none ([INSTALL.md](INSTALL.md#session-tier)).
- **Running without the draft model** (`--drafter none`) is a documented switch. It was checked on the GPU with the
  base weights: drafting with the prediction head gave the same tokens, text and finish reasons as plain
  one-token-at-a-time decoding on every check prompt.
- **The ablit weights** need a Hugging Face account and your own token, and are converted on your machine. Ready in
  this release: `scripts/fetch-weights.sh --weights ablit` downloads the source with your own token,
  `scripts/convert-ablit.sh` converts it and writes the three per-host parts, and `manifests/ablit/` checks a third
  you already have. We ran the shipped conversion on one DGX Spark, and its output matched these manifests file for
  file.
- **Settings outside `config/`** (engine flags passed after `--` to `serve.sh`, `--session-tier off`, or a different
  draft model setting) are not the configuration the published numbers were measured on.
