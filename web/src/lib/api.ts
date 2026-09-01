export type TokenCandidate = {
  token_id: number;
  text: string;
  logit: number;
  probability: number;
};

export type GroupDiagnostic = {
  group: number;
  probability_mass: number;
  top_candidates: TokenCandidate[];
};

export type TokenDiagnostic = {
  index: number;
  token_id: number;
  text: string;
  group: number | null;
  channel_index: number | null;
  block_index: number | null;
  phase: "data" | "formatting" | "tail";
  logit: number | null;
  probability: number | null;
  groups: GroupDiagnostic[];
};

export type EncodeResponse = {
  carrier: string;
  token_count: number;
  fallback_count: number;
  seed: number;
  retry_count: number;
  tokens: TokenDiagnostic[];
};

export type DecodeResponse = {
  payload_base64: string;
  payload_text: string | null;
  corrected_symbols: number;
  tokens: TokenDiagnostic[];
};

export type Settings = {
  key: string;
  groups: number;
  ecc: boolean;
  delta: string;
  temperature: number;
  topP: number;
  tailMaxTokens: number;
  roundtripRetries: number;
  device: string;
};

export function settingsPayload(settings: Settings) {
  return {
    key: settings.key,
    groups: settings.groups,
    ecc: settings.ecc,
    delta: settings.delta.trim() === "" ? null : Number(settings.delta),
    temperature: settings.temperature,
    top_p: settings.topP,
    tail_max_tokens: settings.tailMaxTokens,
    roundtrip_retries: settings.roundtripRetries,
    device: settings.device.trim() || null,
  };
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(body?.detail ?? `API request failed (${response.status})`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Cannot read file"));
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result as ArrayBuffer);
      let binary = "";
      for (let start = 0; start < bytes.length; start += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
      }
      resolve(btoa(binary));
    };
    reader.readAsArrayBuffer(file);
  });
}

export function downloadBase64(value: string, filename: string) {
  const binary = atob(value);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes]));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadText(value: string, filename: string) {
  const url = URL.createObjectURL(new Blob([value], { type: "text/plain;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
