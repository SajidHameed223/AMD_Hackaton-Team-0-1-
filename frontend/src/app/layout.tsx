import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "@/styles/o1.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "O(1) — routed in constant time",
  description:
    "O(1) routes every prompt to the model that earns it: small prompts stay on the local model, hard ones ride out to the cloud.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
