"use client";

import { useId, type CSSProperties } from "react";

/**
 * The opening scene, ported from the "O1 Loading Screen" design:
 * the (1) zooms off, the O does a double-take and rolls after it,
 * the ocean sweeps through and scoops it up, and the reunited mark
 * surfs back in and springs into place. Keyframes live in globals.css.
 */

const MARK = 100; // px — wordmark scale
const LOOP = 7.5; // s — one full story loop

const inf = (name: string, dur: number, extra?: string) =>
  `${name} ${dur}s ${extra ? `${extra} ` : ""}infinite`;

function Ring({ gid, animation }: { gid: string; animation?: string }) {
  return (
    <span style={{ display: "block", animation }}>
      <svg
        width={MARK * 1.16}
        height={MARK * 1.16}
        viewBox="0 0 100 100"
        style={{ display: "block" }}
        aria-hidden
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#22cbe8" />
            <stop offset="0.55" stopColor="#06aece" />
            <stop offset="1" stopColor="#0e7490" />
          </linearGradient>
        </defs>
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke={`url(#${gid})`}
          strokeWidth="13"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}

function Char({ text, ink }: { text: string; ink: boolean }) {
  return (
    <span
      style={{
        fontFamily: "var(--o1-font-display)",
        fontWeight: 500,
        fontSize: MARK * 0.9,
        lineHeight: 1,
        color: ink ? "#10333e" : "#0891b2",
        display: "block",
      }}
    >
      {text}
    </span>
  );
}

function TailChars() {
  return (
    <>
      <Char text="(" ink={false} />
      <Char text="1" ink />
      <Char text=")" ink={false} />
    </>
  );
}

const wordmarkStyle: CSSProperties = {
  gridArea: "1 / 1",
  display: "flex",
  alignItems: "center",
  gap: MARK * 0.07,
};

export function LoadingScreen({ done }: { done: boolean }) {
  const gidA = useId();
  const gidB = useId();

  return (
    <div
      className={`o1-loading${done ? " o1-loading--done" : ""}`}
      role="status"
      aria-label="Loading O(1)"
    >
      <div
        style={{ position: "absolute", inset: 0, overflow: "hidden", pointerEvents: "none" }}
        aria-hidden
      >
        {/* stage: shadow + the two wordmarks */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            paddingTop: "10vh",
          }}
        >
          <div
            style={{
              gridArea: "1 / 1",
              width: MARK * 3.4,
              height: MARK * 0.26,
              borderRadius: "50%",
              background:
                "radial-gradient(closest-side, rgba(16,51,62,0.10), rgba(16,51,62,0))",
              transform: `translateY(${MARK * 0.82}px)`,
              animation: inf("ldg-shadow", LOOP),
            }}
          />
          {/* wordmark A: rests, then splits apart */}
          <div
            style={{
              ...wordmarkStyle,
              animation: inf("ldg-bob", LOOP / 2, "ease-in-out alternate"),
            }}
          >
            <Ring gid={gidA} animation={inf("ldg-o", LOOP)} />
            <span
              style={{
                display: "flex",
                alignItems: "center",
                gap: MARK * 0.02,
                animation: inf("ldg-tail", LOOP),
              }}
            >
              <TailChars />
            </span>
          </div>
          {/* wordmark B: the reunited mark that surfs back in */}
          <div style={{ ...wordmarkStyle, animation: inf("ldg-swoop", LOOP) }}>
            <Ring gid={gidB} />
            <span style={{ display: "flex", alignItems: "center", gap: MARK * 0.02 }}>
              <TailChars />
            </span>
          </div>
        </div>

        {/* speed lines behind the departing (1) */}
        {[-MARK * 0.42, -MARK * 0.05, MARK * 0.3].map((top, i) => (
          <span
            key={`st${i}`}
            style={{
              position: "absolute",
              left: "52%",
              top: `calc(55% + ${top}px)`,
              width: MARK * (0.9 - i * 0.18),
              height: Math.max(4, MARK * 0.05),
              borderRadius: 999,
              background: "#67e0f2",
              opacity: 0,
              transformOrigin: "left center",
              animation: inf("ldg-streak", LOOP),
              animationDelay: `${-i * 0.04}s`,
            }}
          />
        ))}

        {/* foam droplets when the wave grabs the O */}
        {[1, 2, 3].map((i) => (
          <span
            key={`sp${i}`}
            style={{
              position: "absolute",
              left: "50%",
              top: "55%",
              width: 15 - i * 2,
              height: 15 - i * 2,
              borderRadius: "50%",
              background: i === 2 ? "#a9edf8" : "#22cbe8",
              opacity: 0,
              animation: inf(`ldg-splash${i}`, LOOP),
            }}
          />
        ))}

        {/* the wave — drawn once, swept across by ldg-wave */}
        <div
          style={{
            position: "absolute",
            left: 0,
            bottom: "-4vh",
            width: "260vw",
            height: "56vh",
            animation: inf("ldg-wave", LOOP),
          }}
        >
          <div
            style={{
              width: "100%",
              height: "100%",
              transformOrigin: "50% 100%",
              animation: "ldg-wavebob 1.05s ease-in-out infinite alternate",
            }}
          >
            <svg
              width="100%"
              height="100%"
              viewBox="0 0 2800 260"
              preserveAspectRatio="none"
              style={{ display: "block" }}
              aria-hidden
            >
              <defs>
                <linearGradient id="ldgWG" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="#22cbe8" />
                  <stop offset="0.55" stopColor="#06aece" />
                  <stop offset="1" stopColor="#0e7490" />
                </linearGradient>
              </defs>
              <path
                d="M0,260 L0,160 C230,150 470,166 700,158 C930,150 1170,170 1400,158 C1590,148 1740,140 1940,108 C2180,70 2390,58 2550,96 C2650,120 2730,150 2800,162 L2800,260 Z"
                fill="#0e7490"
                opacity="0.22"
                transform="translate(-40,-12)"
              />
              <path
                d="M0,260 L0,178 C240,170 460,188 700,180 C940,172 1180,192 1400,184 C1610,170 1780,152 1970,118 C2210,75 2360,54 2466,66 C2536,74 2576,104 2556,136 C2570,168 2660,192 2800,202 L2800,260 Z"
                fill="url(#ldgWG)"
                opacity="0.96"
              />
              <path
                d="M2040,110 C2220,64 2368,50 2462,62 C2530,71 2562,98 2548,122 C2538,140 2512,140 2504,122 C2498,108 2512,94 2530,98"
                fill="none"
                stroke="#fffdf8"
                strokeWidth="9"
                strokeLinecap="round"
                opacity="0.9"
              />
              <circle cx="2100" cy="94" r="7" fill="#fffdf8" opacity="0.85" />
              <circle cx="2280" cy="58" r="9" fill="#fffdf8" opacity="0.9" />
              <circle cx="2410" cy="56" r="6" fill="#fffdf8" opacity="0.8" />
              <circle cx="360" cy="172" r="5" fill="#fffdf8" opacity="0.5" />
              <circle cx="830" cy="176" r="6" fill="#fffdf8" opacity="0.5" />
              <circle cx="1280" cy="178" r="5" fill="#fffdf8" opacity="0.5" />
            </svg>
          </div>
        </div>
      </div>

      {/* caption + indeterminate waterline */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: "6vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 14,
        }}
      >
        <div className="o1-loading__caption" style={{ textTransform: "none", letterSpacing: "0.01em", fontWeight: 400 }}>
          Catching the fastest wave…
        </div>
        <div className="o1-loading__waterline">
          <span
            className="o1-loading__waterline-fill o1-loading__waterline-fill--indeterminate"
            style={{ display: "block", animation: "ldg-tide 1.7s ease-in-out infinite" }}
          />
        </div>
      </div>
    </div>
  );
}
