# sparkDash adaptation receipt

The pinned author files `DecodeBench.js`, `LlmStreaming.js`, and
`llmPrompts.js` were executed without modification. A separate thin wrapper
supplied only the permitted base URL/port, served model name, and output paths.

Author source commit: `e93fc87d54c8699e98b63a764ab260bf9d446c52`.
The wrapper is `../sparkdash_runner.mjs`; byte-exact HTTP capture is provided by
`../capture_proxy.mjs`.
