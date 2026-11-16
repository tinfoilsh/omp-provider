# Tinfoil provider for omp

Use [Tinfoil](https://tinfoil.sh)'s verifiably-private open models from the
[omp](https://omp.sh) coding agent. Inference runs inside hardware secure
enclaves that even Tinfoil cannot read into.

## Setup

1. Install the plugin:

   ```bash
   omp plugin install @tinfoilsh/omp-provider
   ```

2. Start omp and set your API key:

   ```
   /login
   ```

   Pick **Tinfoil** and paste your key from the
   [Tinfoil dashboard](https://dash.tinfoil.sh). For headless use, set
   `TINFOIL_API_KEY` instead.

3. Pick a Tinfoil model with `/model`.

## How verification works

The plugin uses the [`tinfoil` SDK](https://github.com/tinfoilsh/tinfoil-js) to
verify the inference enclave: it checks the enclave's attestation document and
confirms the running code against the release digest signed in Sigstore. Every
connection is then pinned to the attested TLS key, so only the verified enclave
can read your requests.

Run `/tinfoil` at any time to verify again and print what the verification
proved. When it fails, the report names the check that rejected the enclave.

The plugin also refuses to send requests while `PI_REQ_DEBUG=1`. That setting
makes omp write request bodies to disk, and omp records them before they are
encrypted. Unset the variable, or set `TINFOIL_ALLOW_REQ_DEBUG=1` to accept
plaintext prompt logs and continue.

| Variable | Default | Purpose |
|---|---|---|
| `TINFOIL_API_KEY` | _(none)_ | Your `tk_…` key, for headless workflows. Not needed if you use `/login`. Overrides the stored key. |
| `TINFOIL_ALLOW_REQ_DEBUG` | _(unset)_ | Set to `1` to allow requests while `PI_REQ_DEBUG=1`, which writes your prompts to disk in the clear. |
