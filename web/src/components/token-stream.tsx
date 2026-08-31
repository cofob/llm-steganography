"use client";

import { Tooltip } from "@cofob/design-system-react";
import type { CSSProperties } from "react";

import type { TokenDiagnostic } from "@/lib/api";

import styles from "./token-stream.module.css";

const GROUP_COLORS = [
  "#ef4444",
  "#22c55e",
  "#3b82f6",
  "#f59e0b",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
  "#f97316",
  "#6366f1",
  "#84cc16",
];

function color(group: number) {
  return GROUP_COLORS[group % GROUP_COLORS.length];
}

function visibleToken(value: string) {
  if (value === "\n") return "↵";
  if (value === "\t") return "⇥";
  if (value === " ") return "·";
  if (value.trim() === "") {
    return value.replaceAll(" ", "·").replaceAll("\n", "↵").replaceAll("\t", "⇥");
  }
  return value;
}

function percent(value: number | null) {
  return value === null ? "—" : `${(value * 100).toFixed(value < 0.001 ? 3 : 2)}%`;
}

function number(value: number | null) {
  return value === null ? "—" : value.toFixed(4);
}

function TokenDetails({ token }: { token: TokenDiagnostic }) {
  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipTitle}>
        <span>#{token.index} · token {token.token_id}</span>
        <span>{JSON.stringify(token.text)}</span>
      </div>
      <div className={styles.metaGrid}>
        <span>Phase: {token.phase}</span>
        <span>Group: {token.group ?? "excluded"}</span>
        <span>Channel position: {token.channel_index ?? "—"}</span>
        <span>Block: {token.block_index ?? "—"}</span>
        <span>Selected logit: {number(token.logit)}</span>
        <span>Selection probability: {percent(token.probability)}</span>
      </div>
      {token.groups.length > 0 ? (
        <>
          <div className={styles.tooltipTitle}>Softmax by group · before the steganography mask</div>
          <div className={styles.groupList}>
            {token.groups.map((group) => (
              <div className={styles.groupRow} key={group.group}>
                <span className={styles.groupName}>
                  <span className={styles.swatch} style={{ background: color(group.group) }} />
                  G{group.group}
                </span>
                <span className={styles.barTrack}>
                  <span
                    className={styles.bar}
                    style={{
                      width: `${Math.max(group.probability_mass * 100, 0.35)}%`,
                      background: color(group.group),
                    }}
                  />
                </span>
                <span className={styles.mass}>{percent(group.probability_mass)}</span>
                <span className={styles.candidates}>
                  {group.top_candidates.map((candidate) => (
                    <span key={candidate.token_id}>
                      {JSON.stringify(candidate.text)} #{candidate.token_id} · L {candidate.logit.toFixed(2)} · P {percent(candidate.probability)}{"  "}
                    </span>
                  ))}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div>Logits are unavailable because the text was only decoded or the token is outside the channel.</div>
      )}
    </div>
  );
}

export function TokenStream({
  tokens,
  groups,
  title = "Token map",
}: {
  tokens: TokenDiagnostic[];
  groups: number;
  title?: string;
}) {
  return (
    <section className={styles.panel} aria-label={title}>
      <header className={styles.header}>
        <span className={styles.title}>{title} · {tokens.length}</span>
        <span className={styles.legend}>
          {Array.from({ length: groups }, (_, group) => (
            <span className={styles.legendItem} key={group}>
              <span className={styles.swatch} style={{ background: color(group) }} />
              {group === 0 ? "red" : group === 1 ? "green" : `G${group}`}
            </span>
          ))}
          <span className={styles.legendItem}>·· formatting</span>
          <span className={styles.legendItem}>-- tail</span>
        </span>
      </header>
      {tokens.length === 0 ? (
        <div className={styles.empty}>Tokens appear after an operation.</div>
      ) : (
        <div className={styles.tokens}>
          {tokens.map((token) => {
            const classNames = [styles.token];
            if (token.phase === "formatting") classNames.push(styles.formatting);
            if (token.phase === "tail") classNames.push(styles.tail);
            const tokenStyle = {
              "--token-color": token.group === null ? "#a1a1aa" : color(token.group),
            } as CSSProperties;
            return (
              <Tooltip key={`${token.index}-${token.token_id}`} content={<TokenDetails token={token} />} placement="top" delay={100}>
                <button
                  type="button"
                  className={classNames.join(" ")}
                  style={tokenStyle}
                  aria-label={`Token ${token.index}, group ${token.group ?? "none"}`}
                >
                  {visibleToken(token.text)}
                </button>
              </Tooltip>
            );
          })}
        </div>
      )}
    </section>
  );
}
