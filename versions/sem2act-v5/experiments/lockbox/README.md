# v5 lockbox

Run the generator once after the protocol freeze. It writes qrel-free inputs and
private qrels below the ignored v5 runtime tree. Run the provenance validator
before creating any model bundle or launching any model job.

A failed provenance gate invalidates this lockbox. Do not inspect, patch, or
regenerate individual qrels after the gate.
