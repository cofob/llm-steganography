"use client";

import {
  Alert,
  AppShell,
  Badge,
  BlueLine,
  Button,
  Card,
  ChatThread,
  Checkbox,
  CodeBlock,
  Container,
  EmptyState,
  FileUpload,
  Heading,
  Inline,
  Link,
  Radio,
  RadioGroup,
  Select,
  Separator,
  SkipLink,
  Spinner,
  Stack,
  Tabs,
  Text,
  Textarea,
  TextField,
  ThemeToggle,
  useToast,
} from "@cofob/design-system-react";
import {
  Binary,
  Braces,
  Check,
  Clipboard,
  Download,
  KeyRound,
  MessageSquare,
  Plus,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  apiRequest,
  downloadBase64,
  downloadText,
  fileToBase64,
  settingsPayload,
  type DecodeResponse,
  type EncodeResponse,
  type Settings,
  type TokenDiagnostic,
} from "@/lib/api";

import { TokenStream } from "./token-stream";
import styles from "./steg-app.module.css";

const DEFAULT_SETTINGS: Settings = {
  key: "",
  groups: 2,
  ecc: false,
  delta: "",
  temperature: 0.8,
  topP: 0.95,
  tailMaxTokens: 64,
  roundtripRetries: 3,
  device: "",
  provider: "",
  sglangUrl: "",
  sglangModel: "Qwen/Qwen3.8-27B",
};

type Health = {
  status: string;
  model: string;
  revision: string;
  provider: string;
};

type ChatItem = {
  id: string;
  direction: "outgoing" | "incoming";
  hidden: string;
  carrier: string;
  tokens: TokenDiagnostic[];
};

function ErrorAlert({ error }: { error: string }) {
  return error ? <Alert tone="danger" title="Operation failed">{error}</Alert> : null;
}

function copyText(value: string) {
  return navigator.clipboard.writeText(value);
}

export function StegApp() {
  const { toast } = useToast();
  const [health, setHealth] = useState<Health | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = localStorage.getItem("llm-steg-settings");
      if (stored) {
        try {
          setSettings({
            ...DEFAULT_SETTINGS,
            ...JSON.parse(stored) as Partial<Settings>,
            key: "",
          });
        } catch {
          localStorage.removeItem("llm-steg-settings");
        }
      }
      setHydrated(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (hydrated) {
      localStorage.setItem(
        "llm-steg-settings",
        JSON.stringify({ ...settings, key: "" }),
      );
    }
  }, [hydrated, settings]);

  useEffect(() => {
    apiRequest<Health>("/api/v1/health")
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const updateSetting = useCallback(<K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((current) => ({ ...current, [key]: value }));
  }, []);

  async function makeKey() {
    try {
      const result = await apiRequest<{ key: string }>("/api/v1/key", { method: "POST" });
      updateSetting("key", result.key);
      toast({ title: "New key generated", tone: "success" });
    } catch (error) {
      toast({ title: "Key generation failed", description: String(error), tone: "danger" });
    }
  }

  const tabs = useMemo(() => [
    { id: "encode", label: "Encode", content: <EncodePanel settings={settings} toast={toast} /> },
    { id: "decode", label: "Decode", content: <DecodePanel settings={settings} toast={toast} /> },
    { id: "chat", label: "Chat", content: <ChatPanel settings={settings} toast={toast} /> },
  ], [settings, toast]);

  return (
    <AppShell className={styles.shell}>
      <SkipLink targetId="main">Skip to workspace</SkipLink>
      <header className={styles.siteHeader}>
        <Container size="md">
          <Inline justify="between" align="center">
            <Inline gap="sm" align="center">
              <Binary aria-hidden size={20} />
              <Text as="span" className={styles.brand}>
                <BlueLine>LLM Steganography</BlueLine>
              </Text>
            </Inline>
            <Inline gap="sm" align="center" wrap={false}>
              <span className={`${styles.statusDot} ${health ? styles.statusDotOnline : ""}`} />
              <Text as="span" size="sm" tone="muted" className={styles.headerProtocol}>
                {health ? health.model : "API unavailable"}
              </Text>
              <ThemeToggle aria-label="Change theme" labels={{ light: "Light theme", dark: "Dark theme", system: "System theme" }} />
            </Inline>
          </Inline>
        </Container>
      </header>
      <main id="main">
        <Container size="md" className={styles.main}>
          <Stack gap="lg">
          <Stack gap="sm" className={styles.intro}>
            <Heading level={1} size="xl">Steganography workspace</Heading>
            <Text tone="muted" className={styles.lede}>
              Encode, decode, chat, and inspect the keyed token channel.
            </Text>
          </Stack>

        <div className={styles.workspace}>
          <Card className={styles.workCard} variant="elevated" padding="lg">
            <Tabs items={tabs} defaultValue="encode" label="Mode" />
          </Card>

          <Card className={styles.settings} variant="elevated" padding="lg">
            <div className={styles.cardHeader}>
              <div>
                <Heading level={2} size="md">Channel</Heading>
                <Text size="sm" tone="muted">Shared settings</Text>
              </div>
              <Badge tone={settings.key ? "success" : "warning"}>{settings.key ? "key ready" : "no key"}</Badge>
            </div>
            <Stack gap="md">
              <TextField
                label="Key · base64"
                value={settings.key}
                onChange={(event) => updateSetting("key", event.target.value.trim())}
                placeholder="32 bytes as base64"
                className={styles.keyField}
              />
              <Inline gap="sm" wrap>
                <Button size="sm" variant="secondary" startIcon={KeyRound} onClick={makeKey}>Generate</Button>
                <Button
                  size="sm"
                  variant="ghost"
                  startIcon={Clipboard}
                  disabled={!settings.key}
                  onClick={() => copyText(settings.key).then(() => toast({ title: "Key copied", tone: "success" }))}
                >Copy</Button>
              </Inline>
              <Select
                label="Groups"
                value={settings.groups}
                onChange={(event) => updateSetting("groups", Number(event.target.value))}
              >
                {Array.from({ length: 9 }, (_, index) => index + 2).map((group) => (
                  <option key={group} value={group}>{group} {group === 2 ? "· red / green" : "groups"}</option>
                ))}
              </Select>
              <Checkbox
                label="Reed–Solomon ECC"
                description="Off by default. Adds resilience and tokens."
                checked={settings.ecc}
                onChange={(event) => updateSetting("ecc", event.target.checked)}
              />
              <details className={styles.advanced}>
                <summary>Generation settings</summary>
                <div className={styles.advancedBody}>
                  <Select
                    label="Generation provider"
                    value={settings.provider}
                    onChange={(event) => updateSetting("provider", event.target.value as Settings["provider"])}
                  >
                    <option value="">Server default{health ? ` · ${health.provider}` : ""}</option>
                    <option value="sglang">SGLang API</option>
                    <option value="local">Local weights</option>
                  </Select>
                  {settings.provider === "sglang" ? (
                    <>
                      <TextField label="SGLang API base" hint="OpenAI-compatible /v1 URL" value={settings.sglangUrl} onChange={(event) => updateSetting("sglangUrl", event.target.value)} placeholder="http://host:30000/v1" />
                      <TextField label="Served model" value={settings.sglangModel} onChange={(event) => updateSetting("sglangModel", event.target.value)} />
                    </>
                  ) : null}
                  <TextField
                    label="Threshold δ"
                    hint="Empty: strict mask"
                    inputMode="decimal"
                    value={settings.delta}
                    onChange={(event) => updateSetting("delta", event.target.value)}
                  />
                  <div className={styles.twoColumns}>
                    <TextField label="Temperature" type="number" min="0.01" step="0.05" value={settings.temperature} onChange={(event) => updateSetting("temperature", Number(event.target.value))} />
                    <TextField label="Top P" type="number" min="0.01" max="1" step="0.01" value={settings.topP} onChange={(event) => updateSetting("topP", Number(event.target.value))} />
                    <TextField label="Tail" type="number" min="0" value={settings.tailMaxTokens} onChange={(event) => updateSetting("tailMaxTokens", Number(event.target.value))} />
                    <TextField label="Retries" type="number" min="0" value={settings.roundtripRetries} onChange={(event) => updateSetting("roundtripRetries", Number(event.target.value))} />
                  </div>
                  {settings.provider !== "sglang" ? (
                    <TextField label="Local device" hint="Empty: auto" value={settings.device} onChange={(event) => updateSetting("device", event.target.value)} placeholder="cpu, mps, cuda" />
                  ) : null}
                </div>
              </details>
            </Stack>
          </Card>

        </div>
          <footer className={styles.siteFooter}>
            <Inline gap="md" justify="center" align="center" wrap>
              <Text as="span" size="sm" tone="subtle">
                <Link href="https://cofob.dev/license/" external>
                  License
                </Link>
                {" · "}
                <Link href="https://github.com/cofob/llm-steganography" external>
                  GitHub repository
                </Link>
              </Text>
            </Inline>
          </footer>
          </Stack>
        </Container>
      </main>
    </AppShell>
  );
}

function EncodePanel({ settings, toast }: { settings: Settings; toast: ReturnType<typeof useToast>["toast"] }) {
  const [mode, setMode] = useState("text");
  const [message, setMessage] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("Talk about birds and their behavior.");
  const [seed, setSeed] = useState("");
  const [result, setResult] = useState<EncodeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function encode() {
    if (!settings.key) return setError("Generate or enter a key first.");
    if (mode === "file" && !file) return setError("Select a file.");
    setBusy(true);
    setError("");
    try {
      const payload = mode === "text"
        ? { payload_text: message }
        : { payload_base64: await fileToBase64(file as File) };
      const response = await apiRequest<EncodeResponse>("/api/v1/encode", {
        method: "POST",
        body: JSON.stringify({ ...settingsPayload(settings), ...payload, prompt, seed: seed === "" ? null : Number(seed) }),
      });
      setResult(response);
      toast({ title: "Carrier generated", description: `${response.token_count} tokens`, tone: "success" });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.tabBody}>
      <ErrorAlert error={error} />
      <div className={styles.formGrid}>
        <Stack gap="md" className={styles.formPanel}>
          <RadioGroup name="encode-input" label="Hidden payload" orientation="horizontal">
            <Radio name="encode-input" value="text" label="Text" checked={mode === "text"} onChange={() => setMode("text")} />
            <Radio name="encode-input" value="file" label="File" checked={mode === "file"} onChange={() => setMode("file")} />
          </RadioGroup>
          {mode === "text" ? (
            <Textarea label="Message" rows={5} value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Payload of any size" />
          ) : (
            <FileUpload label="Binary payload" files={file ? [file] : []} maxFiles={1} onFilesChange={(files) => setFile(files[0] ?? null)} prompt="Drop or select a file" />
          )}
          <Textarea label="LLM prompt" rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          <TextField label="Seed" hint="Empty: random" type="number" value={seed} onChange={(event) => setSeed(event.target.value)} />
          <Button startIcon={Sparkles} loading={busy} disabled={busy || !prompt} onClick={encode}>Generate carrier</Button>
        </Stack>
        <Card variant="outlined" padding="md" className={styles.resultPanel}>
          <Stack gap="md">
          <div className={styles.cardHeader}>
            <div><Heading level={3} size="md">Result</Heading><Text size="sm" tone="muted">The carrier continues to the end of the line.</Text></div>
            {busy ? <Spinner label="Generating" /> : result ? <Badge tone="success"><Check size={13} /> ready</Badge> : null}
          </div>
          {result ? (
            <>
              <Inline gap="sm" wrap>
                <Badge>{result.token_count} tokens</Badge>
                <Badge>{result.fallback_count} fallbacks</Badge>
                <Badge>{result.retry_count} retries</Badge>
                <Badge>seed {result.seed}</Badge>
              </Inline>
              <CodeBlock
                className={styles.resultCode}
                code={result.carrier}
                copyable
                copyLabel="Copy carrier"
                copiedLabel="Copied"
              />
              <Inline gap="sm" wrap>
                <Button size="sm" variant="ghost" startIcon={Download} onClick={() => downloadText(result.carrier, "carrier.txt")}>Download</Button>
              </Inline>
            </>
          ) : <EmptyState icon={Braces} title="No carrier yet" description="Token diagnostics are calculated during generation." />}
          </Stack>
        </Card>
      </div>
      <div className={styles.tokenSection}><TokenStream tokens={result?.tokens ?? []} groups={settings.groups} /></div>
    </div>
  );
}

function DecodePanel({ settings, toast }: { settings: Settings; toast: ReturnType<typeof useToast>["toast"] }) {
  const [carrier, setCarrier] = useState("");
  const [files, setFiles] = useState<readonly File[]>([]);
  const [result, setResult] = useState<DecodeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function selectedFiles(next: readonly File[]) {
    setFiles(next);
    if (next[0]) setCarrier(await next[0].text());
  }

  async function decode() {
    if (!settings.key) return setError("Generate or enter a key first.");
    setBusy(true);
    setError("");
    try {
      const response = await apiRequest<DecodeResponse>("/api/v1/decode", {
        method: "POST",
        body: JSON.stringify({ key: settings.key, carrier, ecc: settings.ecc, groups: settings.groups }),
      });
      setResult(response);
      toast({ title: "Payload decoded", tone: "success" });
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.tabBody}>
      <ErrorAlert error={error} />
      <div className={styles.formGrid}>
        <Stack gap="md" className={styles.formPanel}>
          <Textarea label="Carrier text" rows={12} value={carrier} onChange={(event) => setCarrier(event.target.value)} placeholder="Paste the received text without changes" />
          <FileUpload label="Or upload carrier.txt" files={files} maxFiles={1} accept="text/plain" onFilesChange={selectedFiles} prompt="Select a text file" />
          <Button startIcon={ShieldCheck} loading={busy} disabled={busy || !carrier} onClick={decode}>Decode</Button>
        </Stack>
        <Card variant="outlined" padding="md" className={styles.resultPanel}>
          <Stack gap="md">
          <div className={styles.cardHeader}><div><Heading level={3} size="md">Hidden payload</Heading><Text size="sm" tone="muted">Compact format decode result.</Text></div>{result ? <Badge tone="success">decoded</Badge> : null}</div>
          {result ? (
            <>
              <Inline gap="sm" wrap>
                <Badge>{atob(result.payload_base64).length} bytes</Badge>
                <Badge>{result.corrected_symbols} corrected</Badge>
              </Inline>
              <CodeBlock code={result.payload_text ?? `[binary payload · ${atob(result.payload_base64).length} bytes]`} copyable={result.payload_text !== null} copyLabel="Copy payload" copiedLabel="Copied" />
              <Inline gap="sm" wrap>
                <Button size="sm" variant="ghost" startIcon={Download} onClick={() => downloadBase64(result.payload_base64, "payload.bin")}>Download binary</Button>
              </Inline>
            </>
          ) : <EmptyState icon={Binary} title="No decoded payload" description="The compact format has no HMAC. A wrong key can produce random data." />}
          </Stack>
        </Card>
      </div>
      <div className={styles.tokenSection}><TokenStream tokens={result?.tokens ?? []} groups={settings.groups} title="Decoded token groups" /></div>
    </div>
  );
}

function ChatPanel({ settings, toast }: { settings: Settings; toast: ReturnType<typeof useToast>["toast"] }) {
  const [prompt, setPrompt] = useState("Keep the conversation natural. Use the previous message as context and stay on topic.");
  const [mode, setMode] = useState("outgoing");
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionFingerprint, setSessionFingerprint] = useState("");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [selected, setSelected] = useState<ChatItem | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const fingerprint = JSON.stringify({ ...settingsPayload(settings), prompt });

  async function newSession() {
    if (!settings.key) throw new Error("Generate or enter a key first.");
    if (sessionId) apiRequest<void>(`/api/v1/chat/sessions/${sessionId}`, { method: "DELETE" }).catch(() => undefined);
    const response = await apiRequest<{ session_id: string }>("/api/v1/chat/sessions", {
      method: "POST",
      body: JSON.stringify({ ...settingsPayload(settings), prompt }),
    });
    setSessionId(response.session_id);
    setSessionFingerprint(fingerprint);
    setItems([]);
    setSelected(null);
    return response.session_id;
  }

  async function ensureSession() {
    if (sessionId && sessionFingerprint === fingerprint) return sessionId;
    return newSession();
  }

  async function send() {
    if (!input.trim()) return;
    setBusy(true);
    setError("");
    try {
      const id = await ensureSession();
      if (mode === "outgoing") {
        const response = await apiRequest<EncodeResponse>(`/api/v1/chat/sessions/${id}/encode`, { method: "POST", body: JSON.stringify({ message: input }) });
        const item: ChatItem = { id: crypto.randomUUID(), direction: "outgoing", hidden: input, carrier: response.carrier, tokens: response.tokens };
        setItems((current) => [...current, item]);
        setSelected(item);
        toast({ title: "< Message encoded", tone: "success" });
      } else {
        const response = await apiRequest<DecodeResponse>(`/api/v1/chat/sessions/${id}/decode`, { method: "POST", body: JSON.stringify({ carrier: input }) });
        const item: ChatItem = { id: crypto.randomUUID(), direction: "incoming", hidden: response.payload_text ?? `[binary: ${response.payload_base64}]`, carrier: input, tokens: response.tokens };
        setItems((current) => [...current, item]);
        setSelected(item);
        toast({ title: "> Message decoded", tone: "success" });
      }
      setInput("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    setError("");
    try {
      await newSession();
      toast({ title: "New chain created", tone: "success" });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  return (
    <div className={styles.tabBody}>
      <ErrorAlert error={error} />
      <Stack gap="md">
        <Inline justify="between" align="end" wrap>
          <Textarea label="Chat context" rows={2} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          <Button variant="secondary" size="sm" startIcon={Plus} onClick={reset}>New chain</Button>
        </Inline>
        <div className={styles.chatLayout}>
          <Card variant="outlined" padding="none" className={styles.conversation}>
            <div className={styles.messages}>
              {items.length === 0 ? <EmptyState icon={MessageSquare} title="Empty chain" description="The first seed uses only the key. Each next seed also uses the previous carrier hash." /> : (
                <ChatThread
                  label="Steganography chat"
                  messages={items.map((item) => ({
                    id: item.id,
                    author: item.direction === "outgoing" ? "You" : "Peer",
                    own: item.direction === "outgoing",
                    body: (
                      <button type="button" className={styles.chatMessageButton} onClick={() => setSelected(item)}>
                        <span className={styles.messageLabel}>{item.tokens.length} tokens</span>
                        <span className={styles.messageText}>{item.hidden}</span>
                        <span className={styles.carrierPreview}>{item.carrier}</span>
                      </button>
                    ),
                  }))}
                />
              )}
            </div>
            <Separator />
            <div className={styles.composer}>
              <Stack gap="sm">
                <RadioGroup name="chat-mode" label="Direction" orientation="horizontal">
                  <Radio name="chat-mode" value="outgoing" label="< encode" checked={mode === "outgoing"} onChange={() => setMode("outgoing")} />
                  <Radio name="chat-mode" value="incoming" label="> decode" checked={mode === "incoming"} onChange={() => setMode("incoming")} />
                </RadioGroup>
                <Textarea
                  aria-label={mode === "outgoing" ? "Hidden outgoing message" : "Incoming carrier"}
                  rows={mode === "outgoing" ? 3 : 5}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={mode === "outgoing" ? "< hidden message" : "> paste the received carrier"}
                  onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") send();
                  }}
                />
                <Button startIcon={Send} loading={busy} disabled={busy || !input.trim()} onClick={send}>{mode === "outgoing" ? "Encode and add" : "Decode and add"}</Button>
              </Stack>
            </div>
          </Card>
          <Stack gap="md" className={styles.chatInspector}>
            <Card variant="outlined" padding="md">
              <Heading level={3} size="sm">Chain state</Heading>
              <Text size="sm" tone="muted">{items.length} messages</Text>
              <div className={styles.sessionCode}>{sessionId ?? "The session starts with the first message"}</div>
              {sessionId && sessionFingerprint !== fingerprint ? <Alert tone="warning" title="Settings changed">The next message starts a new chain.</Alert> : null}
            </Card>
            {selected ? (
              <>
                <CodeBlock code={selected.carrier} copyable copyLabel="Copy carrier" copiedLabel="Copied" />
                <Inline gap="sm" wrap>
                  <Button size="sm" variant="ghost" startIcon={Download} onClick={() => downloadText(selected.carrier, "chat-carrier.txt")}>Download</Button>
                </Inline>
                <TokenStream tokens={selected.tokens} groups={settings.groups} title="Message tokens" />
              </>
            ) : <Alert tone="info" title="Token inspector">Select a message in the chain.</Alert>}
          </Stack>
        </div>
      </Stack>
    </div>
  );
}
