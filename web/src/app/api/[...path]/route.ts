const DEFAULT_STEG_API_URL = "http://127.0.0.1:8000";

const FORWARDED_REQUEST_HEADERS = [
  "accept",
  "authorization",
  "content-type",
  "traceparent",
  "tracestate",
  "x-request-id",
] as const;

const OMITTED_RESPONSE_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "transfer-encoding",
]);

function upstreamUrl(request: Request): URL {
  const configuredUrl = process.env.STEG_API_URL ?? DEFAULT_STEG_API_URL;
  const upstream = new URL(configuredUrl);
  if (upstream.protocol !== "http:" && upstream.protocol !== "https:") {
    throw new Error("STEG_API_URL must use HTTP or HTTPS");
  }
  if (upstream.username || upstream.password || upstream.search || upstream.hash) {
    throw new Error("STEG_API_URL must not contain credentials, a query, or a fragment");
  }

  const incoming = new URL(request.url);
  const basePath = upstream.pathname.replace(/\/$/, "");
  upstream.pathname = `${basePath}${incoming.pathname}`;
  upstream.search = incoming.search;
  return upstream;
}

function requestHeaders(request: Request): Headers {
  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  headers.set("accept-encoding", "identity");
  return headers;
}

function responseHeaders(upstream: Response): Headers {
  const headers = new Headers();
  upstream.headers.forEach((value, name) => {
    if (!OMITTED_RESPONSE_HEADERS.has(name.toLowerCase())) headers.append(name, value);
  });
  headers.set("cache-control", "no-store");
  return headers;
}

async function proxy(request: Request): Promise<Response> {
  try {
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    const upstream = await fetch(upstreamUrl(request), {
      method: request.method,
      headers: requestHeaders(request),
      body: hasBody && request.body !== null ? await request.arrayBuffer() : undefined,
      cache: "no-store",
      redirect: "manual",
      signal: request.signal,
    });

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders(upstream),
    });
  } catch {
    return Response.json(
      { detail: "LLM API is unavailable" },
      { status: 502, headers: { "cache-control": "no-store" } },
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
