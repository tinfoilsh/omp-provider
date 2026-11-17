import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@oh-my-pi/pi-ai";
// Provider runtime APIs live on providers/* subpaths since pi-ai 18.2.7; the root re-exports them as types only.
import { streamOpenAICompletions } from "@oh-my-pi/pi-ai/providers/openai-completions";
// omp's compiled host bundle omits pi-catalog/build (a plugin import of it falls back to an
// on-disk copy that cannot load); config/model-patch is bundled and rebuilds via buildModel.
import { applyModelPatch } from "@oh-my-pi/pi-coding-agent/config/model-patch";
import type { ExtensionAPI, ProviderModelConfig } from "@oh-my-pi/pi-coding-agent";
// Resolves from this package's node_modules: omp does not rewrite this specifier.
import { RouterHTTPClient } from "tinfoil";

/** What a successful verification proves; the SDK re-exports it only as `current`. */
type Verification = NonNullable<RouterHTTPClient["current"]>;

/**
 * Tinfoil provider for the omp coding agent.
 *
 * Attestation happens in-process via the `tinfoil` SDK: it verifies the
 * enclave's attestation document (the hardware quote, the Sigstore-signed code
 * release, and freshness) and pins every connection to the attested TLS key,
 * so only the verified enclave can read a request.
 *
 * See README.md for setup.
 */

const PROVIDER_ID = "tinfoil";
const ENTRY_TYPE = "tinfoil-report";

interface TinfoilReport {
	trusted: boolean;
	summary: string;
	lines: string[];
}

const HELP_URL = "https://tinfoil.sh/coding-agents";
const DASHBOARD_URL = "https://dash.tinfoil.sh";
const API_KEY_ENV = "TINFOIL_API_KEY";

/**
 * Must stay private. `registerCustomApi` is a global map keyed by this string and
 * `stream()` consults it before any built-in, so reusing `openai-completions`
 * would route every other OpenAI-compatible provider through this extension.
 */
const API_ID = "tinfoil-openai-completions";

/**
 * Stands in until attestation resolves the real origin. `.invalid` is reserved by
 * RFC 2606, so a bug that skips the rewrite fails to connect rather than quietly
 * reaching a real host.
 */
const PLACEHOLDER_BASE_URL = "https://tinfoil.invalid/v1";
const PLACEHOLDER_ORIGIN = new URL(PLACEHOLDER_BASE_URL).origin;

/** omp dumps request bodies to disk when this is exactly "1". */
const REQ_DEBUG_ENV = "PI_REQ_DEBUG";
/** Opt back in to that dump, prompts in the clear, for debugging this provider. */
const REQ_DEBUG_OVERRIDE_ENV = "TINFOIL_ALLOW_REQ_DEBUG";

const DISCOVER_TIMEOUT_MS = 8000;

/** Router discovery alone may try for 30s before falling back; bound the whole attempt. */
const ATTEST_TIMEOUT_MS = 30_000;

const RETRY_COOLDOWN_MS = 30_000;

// =============================================================================
// Verification state
// =============================================================================

/**
 * Anything but "verified" blocks every request. No TTL: a client only exists once
 * verified, and its `fetch` verifies again when the verification expires or the
 * enclave rotates keys.
 */
type VerifyState =
	| { kind: "unverified" }
	| { kind: "verified"; client: RouterHTTPClient }
	| { kind: "failed"; reason: string };

function errorMessage(error: unknown): string {
	const message = error instanceof Error ? error.message : String(error);
	// The SDK retries verification once and appends the first attempt's error;
	// drop it when both attempts failed the same way.
	const attempts = /^(.*) \(first attempt: (.*)\)$/s.exec(message);
	return attempts && attempts[1] === attempts[2] ? attempts[1] : message;
}

/** End a reason with a period, so the hint that follows it starts a new sentence. */
function sentence(reason: string): string {
	return /[.!?]$/.test(reason) ? reason : `${reason}.`;
}

/** Why requests are blocked, as a sentence. */
function reasonOf(state: VerifyState): string {
	return sentence(state.kind === "failed" ? state.reason : "verification did not run");
}

/** Never resolves; rejects when `signal` aborts. */
function aborted(signal: AbortSignal | null | undefined): Promise<never> {
	return new Promise((_, reject) => {
		if (!signal) return;
		const fail = () => reject(signal.reason ?? new Error("aborted"));
		if (signal.aborted) return fail();
		signal.addEventListener("abort", fail, { once: true });
	});
}

// =============================================================================
// Model catalog
// =============================================================================

interface TinfoilApiModel {
	id?: string;
	name?: string;
	type?: string;
	endpoints?: string[];
	context_window?: number;
	max_tokens?: number;
	reasoning?: boolean;
	multimodal?: boolean;
	tool_calling?: boolean;
	pricing?: {
		inputTokenPricePer1M?: number;
		outputTokenPricePer1M?: number;
		requestPrice?: number;
	};
}

/** /v1/models reports no output-token limit; derive a conservative one. */
function deriveMaxTokens(contextWindow: number): number {
	return Math.min(32768, Math.max(4096, Math.floor(contextWindow / 8)));
}

function toModelConfig(raw: TinfoilApiModel): ProviderModelConfig {
	const contextWindow = raw.context_window || 128000;
	return {
		id: raw.id as string,
		name: raw.name ?? (raw.id as string),
		api: API_ID,
		// None of these are OpenAI o-series, and some backends (Kimi K3) reject the
		// `developer` role that reasoning models get by default.
		compat: { supportsDeveloperRole: false },
		reasoning: raw.reasoning === true,
		input: raw.multimodal ? ["text", "image"] : ["text"],
		cost: {
			input: raw.pricing?.inputTokenPricePer1M ?? 0,
			output: raw.pricing?.outputTokenPricePer1M ?? 0,
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow,
		maxTokens: raw.max_tokens ?? deriveMaxTokens(contextWindow),
	};
}

/** A coding agent must never be offered a model that cannot call tools. */
function isUsable(raw: TinfoilApiModel): boolean {
	if (!raw.id) return false;
	if (raw.type && raw.type !== "chat") return false;
	if (raw.endpoints && !raw.endpoints.includes("/v1/chat/completions")) return false;
	if (raw.tool_calling === false) return false;
	return true;
}

/**
 * Load-bearing: on a cold cache a discovery failure would otherwise leave the
 * provider with no models, and omp then points the user at other providers.
 * Stale entries are safe; the guard blocks every request until verification passes.
 * Snapshot of https://inference.tinfoil.sh/v1/models, 2026-09; refresh on release.
 */
const FALLBACK_CATALOG: TinfoilApiModel[] = [
	{
		id: "glm-5-3",
		name: "GLM-5.3",
		context_window: 1048576,
		reasoning: true,
		pricing: { inputTokenPricePer1M: 1.8, outputTokenPricePer1M: 5.75 },
	},
	{
		id: "glm-5-3-flash",
		name: "GLM-5.3 Flash",
		context_window: 1048576,
		reasoning: true,
		multimodal: true,
		pricing: { inputTokenPricePer1M: 0.4, outputTokenPricePer1M: 1.25 },
	},
	{
		id: "deepseek-v4-1-flash",
		name: "DeepSeek V4.1 Flash",
		context_window: 1048576,
		reasoning: true,
		multimodal: true,
		pricing: { inputTokenPricePer1M: 0.65, outputTokenPricePer1M: 1.45 },
	},
	{
		id: "kimi-k3",
		name: "Kimi K3",
		context_window: 262144,
		reasoning: true,
		multimodal: true,
		pricing: { inputTokenPricePer1M: 4, outputTokenPricePer1M: 20 },
	},
	{
		id: "gpt-oss-120b",
		name: "GPT-OSS 120B",
		context_window: 131072,
		reasoning: true,
		pricing: { inputTokenPricePer1M: 0.15, outputTokenPricePer1M: 0.6 },
	},
	{
		id: "gemma4-31b",
		name: "Gemma 4 31B",
		context_window: 262144,
		reasoning: true,
		multimodal: true,
		pricing: { inputTokenPricePer1M: 0.4, outputTokenPricePer1M: 1 },
	},
	{
		id: "llama3-3-70b",
		name: "Llama 3.3 70B",
		context_window: 131072,
		pricing: { inputTokenPricePer1M: 1.75, outputTokenPricePer1M: 2.75 },
	},
];

// =============================================================================
// Rendering
// =============================================================================

/**
 * All three glyphs must stay double-width, or the line jumps when the state flips.
 * None appears in omp's symbol table. "unverified" is neutral, not an error.
 */
const MARKS: Record<VerifyState["kind"], { glyph: string; token: string }> = {
	verified: { glyph: "\u{1F50F}", token: "success" },
	failed: { glyph: "\u{1F6AB}", token: "error" },
	unverified: { glyph: "\u231B", token: "dim" },
};

const shortHash = (value?: string) => (value ? value.replace(/^sha256:/, "").slice(0, 12) : "unknown");



const mark = (state: VerifyState) => MARKS[state.kind].glyph;
const markToken = (state: VerifyState) => MARKS[state.kind].token;



function summaryParts(state: VerifyState, verification?: Verification): { verdict: string; detail: string } {
	if (state.kind === "verified") return { verdict: "Tinfoil verified", detail: shortHash(verification?.codeDigest) };
	if (state.kind === "failed") return { verdict: "Tinfoil unverified", detail: "" };
	return { verdict: "Tinfoil verifying", detail: "" };
}

function summary(state: VerifyState, verification?: Verification): string {
	const { verdict, detail } = summaryParts(state, verification);
	return [verdict, mark(state), detail].filter(Boolean).join(" ");
}

/** A measurement's set registers, one per line, labeled as the verifier names them. */
function measurementLines(label: string, measurement: Record<string, string | undefined>): string[] {
	const { platform, ...registers } = measurement;
	return [
		`  ${label.padEnd(17)}${platform ?? "unknown"}`,
		...Object.entries(registers)
			.filter((pair): pair is [string, string] => Boolean(pair[1]))
			.map(([name, value]) => `    ${name.padEnd(15)}${value}`),
	];
}

const timeOf = (value?: Date) => value?.toISOString() ?? "unknown";

/** The environment silently outranks `/login`, so a surprise 401 needs attributing. */
function credentialSource(envKey: string | undefined): string {
	return envKey ? `${API_KEY_ENV} (environment; overrides /login)` : "stored by /login";
}

function reportLines(
	state: VerifyState,
	verification: Verification | undefined,
	baseUrl: string,
	models: number,
	envKey: string | undefined,
): string[] {
	if (state.kind !== "verified" || !verification) {
		const reason =
			state.kind === "failed" ? state.reason : "no request has triggered verification yet in this session";
		// No base URL: nothing was verified, so there is no enclave URL to show.
		return [
			`Credential: ${credentialSource(envKey)}`,
			`Reason:     ${reason}`,
			"",
			`Run /tinfoil to verify now. See ${HELP_URL}`,
		];
	}

	// The JSON form renders registers as hex and drops facts that were not established.
	const facts = verification.toJSON();
	return [
		"Connection",
		`  Base URL:        ${baseUrl}`,
		`  Enclave host:    ${state.client.enclave || "unknown"}`,
		`  Config repo:     ${facts.configRepo || "unknown"}`,
		`  Models:          ${models}`,
		`  Credential:      ${credentialSource(envKey)}`,
		"",
		"Release",
		`  Tag:             ${facts.codeTag || "unknown"}`,
		`  Digest:          ${facts.codeDigest || "unknown"}`,
		...(facts.runtime ? [`  Runtime:         ${facts.runtime.repo}@${facts.runtime.tag} (${shortHash(facts.runtime.digest)})`] : []),
		...(facts.config ? [`  Config:          ${facts.config.name || "unnamed"} (${shortHash(facts.config.digest)})`] : []),
		"",
		"Attested keys",
		`  TLS fingerprint: ${verification.tlsPublicKeyFP ?? "unknown"}`,
		`  HPKE public key: ${verification.hpkePublicKey ?? "unknown"}`,
		"",
		"Measurements",
		...measurementLines("Code:", facts.codeMeasurement),
		...measurementLines("Enclave:", facts.enclaveMeasurement),
		"",
		"Verifier",
		`  Verifier:        ${facts.metadata.verifier.name} ${facts.metadata.verifier.version}`,
		`  Verified at:     ${timeOf(verification.metadata.verifiedAt)}`,
		`  Fresh until:     ${verification.freshnessExpiresAt ? timeOf(verification.freshnessExpiresAt) : "no expiry"}`,
	];
}

// =============================================================================
// Extension
// =============================================================================

export default function (pi: ExtensionAPI) {
	// Created on first use, so a startup outage cannot stop omp loading. A client
	// only exists once its router has verified.
	let client: RouterHTTPClient | undefined;

	let state: VerifyState = { kind: "unverified" };
	let inFlight: Promise<VerifyState> | undefined;
	let lastAttemptAt = 0;
	let modelCount = FALLBACK_CATALOG.length;
	/** Bumped on every forced re-verification, to retire superseded attempts. */
	let epoch = 0;

	/**
	 * Re-verifies the router we have; if it no longer verifies, discovers again,
	 * which may pick a different router. Never touches `client`: only the
	 * attempt that wins the epoch may install its result.
	 */
	const attempt = async (): Promise<VerifyState> => {
		const signal = AbortSignal.timeout(ATTEST_TIMEOUT_MS);
		try {
			if (client) {
				const existing = client;
				try {
					await existing.reverify({ signal });
					return { kind: "verified", client: existing };
				} catch (error) {
					if (signal.aborted) throw error;
				}
			}
			return { kind: "verified", client: await RouterHTTPClient.create({ signal }) };
		} catch (error) {
			const reason = signal.aborted
				? `enclave verification timed out after ${ATTEST_TIMEOUT_MS}ms`
				: errorMessage(error);
			return { kind: "failed", reason };
		}
	};

	/** Installs a verified client, closing the one it replaces (which failed to verify). */
	const adopt = (next: VerifyState) => {
		if (next.kind !== "verified" || next.client === client) return;
		const previous = client;
		client = next.client;
		void previous?.close();
	};

	/** Concurrent callers share one attempt; failures are cached for the cooldown. */
	const ensureVerified = async (force = false): Promise<VerifyState> => {
		if (force) {
			state = { kind: "unverified" };
			inFlight = undefined;
			epoch += 1;
		} else {
			if (state.kind === "verified") return state;
			if (inFlight) return inFlight;
			if (state.kind === "failed" && Date.now() - lastAttemptAt < RETRY_COOLDOWN_MS) return state;
		}

		// The epoch check retires a superseded attempt: without it a slow success can
		// land after a newer failure and re-open the guard.
		const mine = epoch;
		inFlight = attempt().then((next) => {
			if (mine !== epoch) {
				// Superseded: a client this attempt discovered is nobody's to use, so
				// close it and hand callers the newer attempt's result instead.
				if (next.kind === "verified" && next.client !== client) void next.client.close();
				return inFlight ?? state;
			}
			adopt(next);
			state = next;
			lastAttemptAt = Date.now();
			inFlight = undefined;
			return next;
		});
		return inFlight;
	};

	const requestDebugBlocked = (): boolean =>
		process.env[REQ_DEBUG_ENV] === "1" && process.env[REQ_DEBUG_OVERRIDE_ENV] !== "1";

	const envApiKey = (): string | undefined => process.env[API_KEY_ENV]?.trim() || undefined;

	/**
	 * Scheme and host only, so nothing in the path can move the request elsewhere;
	 * the placeholder's `/v1` path is kept, and `client.baseURL` is the router's
	 * root. The SDK still refuses any origin but the verified enclave, so a bad
	 * rewrite fails closed.
	 */
	const toEnclave = (input: RequestInfo | URL, base: string): RequestInfo | URL => {
		const swap = (raw: string): string => {
			// The SDK's fetch takes only absolute URLs; resolve any relative path here.
			const url = new URL(raw, base);
			if (url.origin !== PLACEHOLDER_ORIGIN) return url.toString();
			const enclave = new URL(base);
			url.protocol = enclave.protocol;
			url.host = enclave.host;
			return url.toString();
		};

		if (typeof input === "string" || input instanceof URL) return swap(String(input));
		if (input instanceof Request) {
			const swapped = swap(input.url);
			return swapped === input.url ? input : new Request(swapped, input);
		}
		return input;
	};

	/**
	 * The one place every byte leaves this machine: discovery and inference both route
	 * through here, so a new call site cannot bypass these checks.
	 */
	const guardedFetch: typeof fetch = async (input, init) => {
		if (requestDebugBlocked()) {
			throw new Error(
				`Tinfoil: refusing to send this request. ${REQ_DEBUG_ENV}=1 makes omp write request bodies to disk, ` +
					`and it records them before Tinfoil encrypts them, so your prompts would be stored in the clear. ` +
					`Unset ${REQ_DEBUG_ENV}, or set ${REQ_DEBUG_OVERRIDE_ENV}=1 to accept that and continue. See ${HELP_URL}`,
			);
		}

		// Taken before verifying, so a forced re-verification that lands while this
		// request waits still counts as one that may have closed its client.
		const mine = epoch;
		// Attempts are shared between callers, so race this caller's signal rather
		// than aborting an attempt others are waiting on.
		const current = await Promise.race([ensureVerified(), aborted(init?.signal)]);
		if (current.kind !== "verified") {
			throw new Error(
				`Tinfoil: refusing to send this request. Enclave verification failed: ${reasonOf(current)} ` +
					`Run /tinfoil to retry, or see ${HELP_URL}`,
			);
		}

		const send = (verified: RouterHTTPClient) => verified.fetch(toEnclave(input, verified.baseURL), init);
		try {
			return await send(current.client);
		} catch (error) {
			// A forced re-verification may replace and close the client mid-flight;
			// retry once so /tinfoil cannot kill an in-flight turn. Not after the
			// caller gave up, though.
			if (mine === epoch || init?.signal?.aborted) throw error;
			const retry = await Promise.race([ensureVerified(), aborted(init?.signal)]);
			if (retry.kind !== "verified") throw error;
			return await send(retry.client);
		}
	};

	/**
	 * The compat engine matches api ids exactly, so our private API_ID resolves to
	 * `compat: undefined` and the OpenAI provider then dereferences it. Rebuilding
	 * under the real id reuses that resolution instead of hand-maintaining ~60 fields.
	 */
	const asCompletionsModel = (model: Model<Api>): Model<"openai-completions"> => {
		// Drop the fields buildModel derives, so they are re-derived for the real id. applyModelPatch's
		// toModelSpec already drops `compat` and `supportsComputerUseConfig` and keeps `compatConfig`.
		const {
			identity: _identity,
			requiresGlyphTokenization: _glyph,
			requiresCursorToolSchemaProjection: _cursor,
			requiresToolResultImageHoisting: _hoist,
			supportsAssistantPrefill: _prefill,
			...spec
		} = model;

		// Uncached: keying by id would pin the first build past a discovery refresh.
		// An empty patch just rebuilds the model under the real api id.
		return applyModelPatch({ ...spec, api: "openai-completions" } as Model<Api>, {}, "merge") as Model<"openai-completions">;
	};

	/**
	 * Throwing leaves omp on the static catalog. omp runs this at session start
	 * whatever the active model, so skip it without a key: otherwise installing the
	 * plugin would attest for people who never select a Tinfoil model.
	 */
	const discoverModels = async (apiKey?: string): Promise<ProviderModelConfig[]> => {
		if (!apiKey) return FALLBACK_CATALOG.map(toModelConfig);

		const response = await guardedFetch(`${PLACEHOLDER_BASE_URL}/models`, {
			headers: { Authorization: `Bearer ${apiKey}` },
			signal: AbortSignal.timeout(DISCOVER_TIMEOUT_MS),
		});
		if (!response.ok) throw new Error(`HTTP ${response.status} from /v1/models`);
		const body = (await response.json()) as { data?: TinfoilApiModel[] };
		const models = (body.data ?? []).filter(isUsable).map(toModelConfig);
		modelCount = models.length;
		return models;
	};

	pi.registerProvider(PROVIDER_ID, {
		baseUrl: PLACEHOLDER_BASE_URL,
		api: API_ID,
		/**
		 * Pin only when set. `resolveConfigValue` falls back to the literal string when
		 * the variable is unset, and a config-pinned key outranks stored credentials —
		 * so pinning unconditionally sends "TINFOIL_API_KEY" as the bearer token and
		 * makes `/login` impossible.
		 */
		...(envApiKey() ? { apiKey: API_KEY_ENV } : {}),
		authHeader: true,
		models: FALLBACK_CATALOG.map(toModelConfig),
		fetchDynamicModels: discoverModels,
		/**
		 * Tinfoil issues long-lived keys, so returning a bare string tells omp to store
		 * one. Registering `oauth` at all is what puts Tinfoil in `/login`.
		 */
		oauth: {
			name: "Tinfoil",
			async login(callbacks) {
				callbacks.onAuth({
					url: DASHBOARD_URL,
					instructions: "Create an API key in the Tinfoil dashboard, then paste it here",
				});

				const entered = await callbacks.onPrompt({
					message: "Paste your Tinfoil API key",
					placeholder: "tk_...",
				});
				if (callbacks.signal?.aborted) throw new Error("Tinfoil login cancelled");

				const apiKey = entered.trim();
				if (!apiKey) throw new Error("A Tinfoil API key is required");

				// Verify the enclave, not the key: Tinfoil has no key-check endpoint, and an
				// invalid key surfaces as a 401 on first use. Forced, so a stale failure in
				// the cooldown cannot fail a login the user just fixed the network for.
				callbacks.onProgress?.("Verifying the Tinfoil enclave...");
				const verified = await ensureVerified(true);
				if (verified.kind !== "verified") {
					throw new Error(`Tinfoil enclave verification failed: ${reasonOf(verified)} See ${HELP_URL}`);
				}

				return apiKey;
			},
		},
		/**
		 * `fetch` is applied last on purpose: omp pre-fills `options.fetch` with its own
		 * transport, and letting it through would send bodies outside the attested channel.
		 */
		streamSimple: (
			model: Model<Api>,
			context: Context,
			options?: SimpleStreamOptions,
		): AssistantMessageEventStream => {
			// pi-ai resolves an ApiKeyResolver before dispatching, so this is always a string.
			const { apiKey, ...rest } = options ?? {};
			return streamOpenAICompletions(asCompletionsModel(model), context, {
				...rest,
				apiKey: typeof apiKey === "string" ? apiKey : undefined,
				fetch: guardedFetch,
			});
		},
	});

	type StatusContext = {
		model?: { provider?: string };
		ui: {
			// Optional: headless modes have no editor to hang a widget on.
			setWidget?(
				key: string,
				content: string[] | undefined,
				options?: { placement?: "aboveEditor" | "belowEditor" },
			): void;
			theme?: { fg(token: string, text: string): string };
			notify(message: string, type?: "info" | "warning" | "error"): void;
		};
	};

	const usesTinfoil = (ctx: StatusContext) => ctx.model?.provider === PROVIDER_ID;

	/** Read at render time so a self-recovered rotation shows up. */
	const verificationOf = (): Verification | undefined => (state.kind === "verified" ? state.client.current : undefined);

	const baseUrlOf = (): string => (state.kind === "verified" ? `${state.client.baseURL}v1` : PLACEHOLDER_BASE_URL);

	/**
	 * Hidden unless Tinfoil is active, or it claims a guarantee that does not apply.
	 * A widget, not `setStatus`: omp runs status text through `Bun.stripANSI`.
	 */
	const showStatus = (ctx: StatusContext) => {
		const theme = ctx.ui.theme;
		if (!ctx.ui.setWidget || !theme) return;
		if (!usesTinfoil(ctx)) {
			ctx.ui.setWidget(PROVIDER_ID, undefined);
			return;
		}

		const { verdict, detail } = summaryParts(state, verificationOf());
		const line = [
			theme.fg(markToken(state), mark(state)),
			theme.fg("dim", verdict),
			detail ? theme.fg("dim", detail) : "",
		]
			.filter(Boolean)
			.join(" ");

		// Above the editor, right on top of omp's status line, so it reads as part of it.
		ctx.ui.setWidget(PROVIDER_ID, [line], { placement: "aboveEditor" });
	};

	const warning = (): { message: string; level: "warning" | "error" } | undefined => {
		if (requestDebugBlocked()) {
			return {
				message:
					`Tinfoil: ${REQ_DEBUG_ENV}=1 writes request bodies to disk before they are encrypted. ` +
					`Requests are blocked. Unset it, or set ${REQ_DEBUG_OVERRIDE_ENV}=1 to accept plaintext prompt logs.`,
				level: "error",
			};
		}
		if (state.kind === "failed") {
			return {
				message:
					`Tinfoil: enclave verification failed: ${reasonOf(state)} ` +
					`Requests are blocked. Run /tinfoil for the full report. See ${HELP_URL}`,
				level: "error",
			};
		}
		return undefined;
	};

	pi.registerCommand(PROVIDER_ID, {
		description: "Verify the Tinfoil enclave and show the verification document",
		handler: async (_args, ctx) => {
			await ensureVerified(true);

			const verification = verificationOf();
			const trusted = state.kind === "verified";
			const lines = reportLines(state, verification, baseUrlOf(), modelCount, envApiKey());
			const head = summary(state, verification);

			showStatus(ctx);
			// Notify, not a transcript message: omp turns every custom message into a
			// `developer` message, so a rendered report would spend context each time.
			ctx.ui.notify([head, ...lines].join("\n"), trusted ? "info" : "error");
			// Journal only: kept in the session file without entering the conversation.
			pi.appendEntry<TinfoilReport>(ENTRY_TYPE, { trusted, summary: head, lines });
		},
	});

	// omp has no model_select event, so refresh at the points the model can change.
	pi.on("turn_start", async (_event, ctx) => {
		showStatus(ctx as unknown as StatusContext);
	});

	pi.on("session_start", async (_event, ctx) => {
		const status = ctx as unknown as StatusContext;
		showStatus(status);
		if (!usesTinfoil(status)) return;

		// One message per session; a stack of warnings trains users to ignore them.
		const notifyProblem = () => {
			const problem = warning();
			if (problem) status.ui.notify(problem.message, problem.level);
		};

		// Known without network work, so report before attesting.
		if (requestDebugBlocked()) {
			notifyProblem();
			return;
		}

		// Background: never awaited, or session start blocks on the network. Without it
		// the indicator sits at "verifying" until the first message and reads as a fault.
		void ensureVerified()
			.then((result) => {
				showStatus(status);
				if (result.kind === "failed") notifyProblem();
			})
			// Extensions share the session's process; an unhandled rejection kills it.
			.catch(() => {});
	});
}
