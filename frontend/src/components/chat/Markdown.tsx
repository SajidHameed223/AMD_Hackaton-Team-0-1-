"use client";

import { memo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { CodeBlock } from "./CodeBlock";

/**
 * While a reply is still streaming, unterminated ``` fences or $$ blocks
 * would flash as plain text; close them so partial markdown renders steady.
 */
export function closeDanglingMarkdown(src: string): string {
  let out = src;
  const fences = (out.match(/```/g) ?? []).length;
  if (fences % 2 === 1) out += "\n```";
  const mathBlocks = (out.match(/\$\$/g) ?? []).length;
  if (mathBlocks % 2 === 1) out += "$$";
  return out;
}

function extractText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object" && "props" in node) {
    return extractText(
      (node as { props: { children?: ReactNode } }).props.children,
    );
  }
  return "";
}

export const Markdown = memo(function Markdown({
  children,
  streaming = false,
}: {
  children: string;
  streaming?: boolean;
}) {
  const source = streaming ? closeDanglingMarkdown(children) : children;
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        pre({ children }) {
          // fenced code: unwrap the <pre>, CodeBlock brings its own chrome
          return <>{children}</>;
        },
        code({ className, children }) {
          const lang = /language-(\w+)/.exec(className ?? "")?.[1];
          const text = extractText(children).replace(/\n$/, "");
          if (!lang && !text.includes("\n")) {
            return <code>{children}</code>;
          }
          return <CodeBlock code={text} language={lang} />;
        },
      }}
    >
      {source}
    </ReactMarkdown>
  );
});
