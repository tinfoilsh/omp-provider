import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@oh-my-pi/pi-ai";
// Provider runtime APIs live on providers/* subpaths since pi-ai 18.2.7; the root re-exports them as types only.
import { streamOpenAICompletions } from "@oh-my-pi/pi-ai/providers/openai-completions";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import type { ExtensionAPI, ProviderModelConfig } from "@oh-my-pi/pi-coding-agent";
// Resolves from this package's node_modules: omp does not rewrite this specifier.
import { SecureClient, type VerificationDocument } from "tinfoil";

/**
 * Tinfoil provider for the omp coding agent.
 *
 * Attestation and encryption happen in-process via the `tinfoil` SDK: it
 * verifies the SEV-SNP report against a pinned AMD root key (ARK to ASK to
 * VCEK, plus the report signature), checks the Sigstore-signed code digest,
 * and encrypts every request body end-to-end with HPKE. It does not check
 * certificate revocation and covers SEV-SNP only.
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

/** Attestation has no timeout of its own, so an unreachable endpoint would hang the turn forever. */
const ATTEST_TIMEOUT_MS = 30_000;

const RETRY_COOLDOWN_MS = 30_000;

// Drift is a warning, not a failure: the attestation behind the fields still holds.
const KNOWN_SCHEMA_VERSION = 1;

// =============================================================================
// Verification state
// =============================================================================

/**
 * Anything but "verified" blocks every request. No TTL: verification binds to the
 * live connection and `SecureClient.fetch` re-attests when the enclave rotates keys.
 */
type VerifyState =
	| { kind: "unverified" }
	| { kind: "verified" }
	| { kind: "failed"; reason: string };

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function reasonOf(state: VerifyState): string {
	return state.kind === "failed" ? state.reason : "verification did not run";
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

/** Reject if `work` outlasts `ms`. The underlying work is not cancellable. */
async function withDeadline<T>(work: Promise<T>, ms: number, label: string): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		return await Promise.race([
			work,
			new Promise<never>((_, reject) => {
				timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
			}),
		]);
	} finally {
		if (timer) clearTimeout(timer);
	}
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



function summaryParts(state: VerifyState, document?: VerificationDocument): { verdict: string; detail: string } {
	if (state.kind === "verified") return { verdict: "Tinfoil verified", detail: shortHash(document?.releaseDigest) };
	if (state.kind === "failed") return { verdict: "Tinfoil unverified", detail: "" };
	return { verdict: "Tinfoil verifying", detail: "" };
}

function summary(state: VerifyState, document?: VerificationDocument): string {
	const { verdict, detail } = summaryParts(state, document);
	return [verdict, mark(state), detail].filter(Boolean).join(" ");
}

/** Ordered as the verifier performs the steps. */
function stepLines(document: VerificationDocument): string[] {
	const steps = document.steps;
	if (!steps) return [];
	const entries: Array<[string, { status?: string; error?: string } | undefined]> = [
		["Fetch digest", steps.fetchDigest],
		["Verify code", steps.verifyCode],
		["Verify enclave", steps.verifyEnclave],
		["Compare measurements", steps.compareMeasurements],
		["Verify certificate", steps.verifyCertificate],
		["Other", steps.otherError],
	];
	return entries.flatMap(([name, step]) =>
		step ? [`  ${name.padEnd(22)}${step.status ?? "unknown"}${step.error ? `: ${step.error}` : ""}`] : [],
	);
}

/** The environment silently outranks `/login`, so a surprise 401 needs attributing. */
function credentialSource(envKey: string | undefined): string {
	return envKey ? `${API_KEY_ENV} (environment; overrides /login)` : "stored by /login";
}

function reportLines(
	state: VerifyState,
	document: VerificationDocument | undefined,
	baseUrl: string,
	models: number,
	envKey: string | undefined,
): string[] {
	if (state.kind !== "verified" || !document) {
		const reason =
			state.kind === "failed" ? state.reason : "no request has triggered verification yet in this session";
		return [
			`Base URL:   ${baseUrl}`,
			`Credential: ${credentialSource(envKey)}`,
			`Reason:     ${reason}`,
			"",
			`Run /tinfoil to verify now. See ${HELP_URL}`,
		];
	}

	const enclave = document.enclaveMeasurement ?? {};
	return [
		"Connection",
		`  Base URL:        ${baseUrl}`,
		`  Enclave host:    ${document.enclaveHost || "unknown"}`,
		`  Router endpoint: ${document.selectedRouterEndpoint || "unknown"}`,
		`  Config repo:     ${document.configRepo || "unknown"}`,
		`  Models:          ${models}`,
		`  Credential:      ${credentialSource(envKey)}`,
		"",
		"Release",
		`  Tag:             ${document.releaseTag ?? "unknown"}`,
		`  Digest:          ${document.releaseDigest || "unknown"}`,
		`  Code print:      ${document.codeFingerprint || "unknown"}`,
		`  Enclave print:   ${document.enclaveFingerprint || "unknown"}`,
		"",
		"Attested keys",
		`  TLS public key:  ${document.tlsPublicKey || "unknown"}`,
		`  TLS fingerprint: ${enclave.tlsPublicKeyFingerprint ?? "unknown"}`,
		`  HPKE public key: ${document.hpkePublicKey || enclave.hpkePublicKey || "unknown"}`,
		"",
		"Measurements",
		`  Code:            ${document.codeMeasurement?.type || "unknown"}`,
		...(document.codeMeasurement?.registers ?? []).map((register) => `                   ${register}`),
		`  Enclave:         ${enclave.measurement?.type ?? "unknown"}`,
		...(enclave.measurement?.registers ?? []).map((register) => `                   ${register}`),
		"",
		"Verification steps",
		...stepLines(document),
		"",
		"Verifier",
		`  Verifier:        ${document.verifier?.name ?? "unknown"} ${document.verifier?.version ?? ""}`.trimEnd(),
		`  Verified:        ${document.securityVerified === true ? "yes" : "(SDK did not mark this document verified)"}`,
		`  Verified at:     ${document.verifiedAt ?? "unknown"}`,
	];
}

// =============================================================================
// Extension
// =============================================================================

export default function (pi: ExtensionAPI) {
	// Constructor does no network, so a startup outage cannot stop omp loading.
	const secureClient = new SecureClient();

	let state: VerifyState = { kind: "unverified" };
	let inFlight: Promise<VerifyState> | undefined;
	let lastAttemptAt = 0;
	let modelCount = FALLBACK_CATALOG.length;
	/** Bumped on every forced re-verification, to retire superseded attempts. */
	let epoch = 0;

	const attempt = async (): Promise<VerifyState> => {
		try {
			await withDeadline(secureClient.ready(), ATTEST_TIMEOUT_MS, "enclave verification");
			return { kind: "verified" };
		} catch (error) {
			return { kind: "failed", reason: errorMessage(error) };
		}
	};

	/** Concurrent callers share one attempt; failures are cached for the cooldown. */
	const ensureVerified = async (force = false): Promise<VerifyState> => {
		if (force) {
			secureClient.reset();
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
			if (mine !== epoch) return next;
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
	 * Origin only — the placeholder and `getBaseURL()` share the `/v1` path. The SDK
	 * still refuses any origin but the verified enclave, so a bad rewrite fails closed.
	 */
	const toEnclave = (input: RequestInfo | URL): RequestInfo | URL => {
		const base = secureClient.getBaseURL();
		if (!base) return input;

		const swap = (raw: string): string => {
			let url: URL;
			try {
				url = new URL(raw);
			} catch {
				// Relative path: the SDK resolves it against the enclave itself.
				return raw;
			}
			if (url.origin !== PLACEHOLDER_ORIGIN) return raw;
			return new URL(url.pathname + url.search + url.hash, base).toString();
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

		// Attestation is not cancellable, so race the signal or Ctrl-C does nothing.
		const current = await Promise.race([ensureVerified(), aborted(init?.signal)]);
		if (current.kind !== "verified") {
			throw new Error(
				`Tinfoil: refusing to send this request. Enclave verification failed: ${reasonOf(current)} ` +
					`Run /tinfoil to retry, or see ${HELP_URL}`,
			);
		}

		const send = () => secureClient.fetch(toEnclave(input), init);
		const mine = epoch;
		try {
			return await send();
		} catch (error) {
			// A forced re-verification resets the transport mid-flight; retry once so
			// /tinfoil cannot kill an in-flight turn.
			if (mine === epoch) throw error;
			const retry = await ensureVerified();
			if (retry.kind !== "verified") throw error;
			return await send();
		}
	};

	/**
	 * The compat engine matches api ids exactly, so our private API_ID resolves to
	 * `compat: undefined` and the OpenAI provider then dereferences it. Rebuilding
	 * under the real id reuses that resolution instead of hand-maintaining ~60 fields.
	 */
	const asCompletionsModel = (model: Model<Api>): Model<"openai-completions"> => {
		// Drop the fields buildModel derives, so this is a spec and not a built model.
		const {
			compat: _compat,
			identity: _identity,
			compatConfig,
			requiresGlyphTokenization: _glyph,
			requiresCursorToolSchemaProjection: _cursor,
			requiresToolResultImageHoisting: _hoist,
			supportsAssistantPrefill: _prefill,
			supportsComputerUseConfig: _computer,
			...spec
		} = model;

		// Uncached: keying by id would pin the first build past a discovery refresh.
		return buildModel({
			...spec,
			api: "openai-completions",
			compat: compatConfig,
		} as Parameters<typeof buildModel<"openai-completions">>[0]);
	};

	/**
	 * Throwing leaves omp on the static catalog. omp runs this at session start
	 * whatever the active model, so skip it without a key: otherwise installing the
	 * plugin would attest for people who never select a Tinfoil model.
	 */
	const discoverModels = async (apiKey?: string): Promise<ProviderModelConfig[]> => {
		if (!apiKey) return FALLBACK_CATALOG.map(toModelConfig);

		const response = await guardedFetch("/v1/models", {
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

	/**
	 * Read at render time so a self-recovered rotation shows up. Returned even on
	 * failure: the partial document's per-step errors are the point of /tinfoil.
	 */
	const documentOf = (): VerificationDocument | undefined => {
		try {
			return secureClient.getVerificationDocument();
		} catch {
			return undefined;
		}
	};

	const baseUrlOf = (): string => secureClient.getBaseURL() ?? PLACEHOLDER_BASE_URL;

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

		const { verdict, detail } = summaryParts(state, documentOf());
		const line = [
			theme.fg(markToken(state), mark(state)),
			theme.fg("dim", verdict),
			detail ? theme.fg("dim", detail) : "",
		]
			.filter(Boolean)
			.join(" ");

		ctx.ui.setWidget(PROVIDER_ID, [line], { placement: "belowEditor" });
	};

	// Warn (do not fail) if the document layout drifts ahead of this extension.
	const schemaNote = (): string | undefined => {
		if (state.kind !== "verified") return undefined;
		const version = secureClient.getVerificationDocument().schemaVersion;
		return version === KNOWN_SCHEMA_VERSION
			? undefined
			: `the SDK reports document schema ${version ?? "(none)"}, and this extension knows ${KNOWN_SCHEMA_VERSION}. ` +
					`Some fields in /tinfoil may read as "unknown". Update the extension.`;
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
					`Tinfoil: enclave verification failed: ${state.reason} ` +
					`Requests are blocked. Run /tinfoil for the full report. See ${HELP_URL}`,
				level: "error",
			};
		}
		const note = schemaNote();
		if (note) return { message: `Tinfoil: ${note}`, level: "warning" };
		return undefined;
	};

	pi.registerCommand(PROVIDER_ID, {
		description: "Verify the Tinfoil enclave and show the verification document",
		handler: async (_args, ctx) => {
			await ensureVerified(true);

			const document = documentOf();
			const trusted = state.kind === "verified";
			const lines = reportLines(state, document, baseUrlOf(), modelCount, envApiKey());
			const head = summary(state, document);

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
