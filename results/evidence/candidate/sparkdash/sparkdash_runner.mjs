#!/usr/bin/env node
/** Thin output-path/base-URL/model wrapper around pinned sparkDash DecodeBench. */

import fs from "fs";
import path from "path";

function argsOf(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) out[argv[i].replace(/^--/, "")] = argv[i + 1];
  return out;
}
const args = argsOf(process.argv);
if (!args.source || !args.out || !args.host || !args.port || !args.model) {
  console.error("usage: sparkdash_runner.mjs --source DecodeBench.js --out DIR --host HOST --port PORT --model MODEL");
  process.exit(2);
}
const out = path.resolve(args.out);
fs.mkdirSync(out, { recursive: true });
process.env.BENCH_HISTORY_PATH = path.join(out, "sparkdash-history.json");
process.env.BENCH_ACTIVE_PATH = path.join(out, "sparkdash-active.json");
const { DecodeBenchManager } = await import(new URL(`file://${path.resolve(args.source)}`));
const manager = new DecodeBenchManager(process.env.BENCH_HISTORY_PATH, process.env.BENCH_ACTIVE_PATH);
const types = ["structured", "prose", "code", "json"];
const jobs = [];
for (const promptType of types) {
  const started = manager.start({
    sparkId: "release",
    lanIp: args.host,
    port: Number(args.port),
    modelId: args.model,
    concurrencies: [1, 2, 4],
    maxTokens: 400,
    promptType,
    debug: true,
  });
  let job;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    job = manager.getJob(started.benchId);
    if (job.status !== "running") break;
  }
  jobs.push(job);
  console.log(JSON.stringify({ promptType, status: job.status, error: job.error, results: job.results.map((r) => ({ concurrency: r.concurrency, streamsOk: r.streamsOk, meanDecodeTps: r.meanDecodeTps, aggregateDecodeTps: r.aggregateDecodeTps, meanTtftMs: r.meanTtftMs })) }));
  if (job.status !== "completed" || job.error || job.results.some((r) => r.error || r.streamsFailed)) break;
}
const record = {
  schema_version: 1,
  source: path.resolve(args.source),
  base_url_adaptation: `http://${args.host}:${args.port}`,
  model_adaptation: args.model,
  prompt_type_order: types,
  concurrencies: [1, 2, 4],
  max_tokens: 400,
  jobs,
};
fs.writeFileSync(path.join(out, "SPARKDASH-RESULT.json"), JSON.stringify(record, null, 2) + "\n");
const valid = jobs.length === types.length && jobs.every((job) => job.status === "completed" && !job.error && job.results.length === 3 && job.results.every((row) => !row.error && row.streamsFailed === 0));
process.exit(valid ? 0 : 1);
