import React from "react";

interface ListItem {
  text: string;
  checked?: boolean;
}

interface Block {
  type: "heading" | "paragraph" | "blockquote" | "code" | "list" | "table" | "hr";
  level?: number;
  text?: string;
  lines?: string[];
  language?: string;
  code?: string;
  ordered?: boolean;
  items?: ListItem[];
  headers?: string[];
  alignments?: ("left" | "center" | "right")[];
  rows?: string[][];
}

interface MarkdownReaderProps {
  content: string;
  className?: string;
}

function getSafeLinkTarget(url: string) {
  const trimmed = url.trim();
  const isExternal = /^(https?:)?\/\//i.test(trimmed);
  const isSafe =
    isExternal ||
    /^mailto:/i.test(trimmed) ||
    trimmed.startsWith("/") ||
    trimmed.startsWith("#") ||
    trimmed.startsWith("./") ||
    trimmed.startsWith("../");

  return isSafe ? { href: trimmed, isExternal } : null;
}

function normalizeArchiveText(markdown: string) {
  const lines = markdown.split(/\r?\n/);
  const cleaned: string[] = [];
  let inFence = false;

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      inFence = !inFence;
      cleaned.push(line);
      continue;
    }

    if (!inFence) {
      // Some archived documents include LightRAG graph metadata as standalone
      // marker lines, rendered with block-element sentinels such as
      // `█entity█["company","Opendorse","..."]█`. These are index artifacts,
      // not document prose, so keep them out of the reading view.
      const withoutBlockMarkers = line.replace(/[\u2580-\u259f]/g, "").trim();
      if (/^(entity|relationship)\s*\[.*\]\s*$/i.test(withoutBlockMarkers)) {
        continue;
      }
    }

    cleaned.push(line);
  }

  return cleaned.join("\n");
}

/**
 * Parses a string of inline markdown formatting (bold, italic, code, links, autolinks)
 * and returns an array of React nodes. Support nested parsing inside bold/italic blocks.
 */
function parseInline(text: string): React.ReactNode[] {
  const result: React.ReactNode[] = [];
  // Regex matches:
  // 1: Bold (**text** or __text__)
  // 3: Italic (*text* or _text_)
  // 5: Inline code (`code`)
  // 6: Link text, 7: Link URL ([text](url))
  // 8: Autolink (http/https URL)
  const regex = /(\*\*|__)(.*?)\1|(\*|_)(.*?)\3|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/g;

  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      result.push(text.slice(lastIndex, match.index));
    }

    const [
      ,
      boldMarker, boldText,
      italicMarker, italicText,
      codeText,
      linkText, linkUrl,
      autoUrl
    ] = match;

    if (boldText !== undefined) {
      result.push(
        <strong key={`b-${key++}`} className="font-semibold text-ink">
          {parseInline(boldText)}
        </strong>
      );
    } else if (italicText !== undefined) {
      result.push(
        <em key={`i-${key++}`} className="italic">
          {parseInline(italicText)}
        </em>
      );
    } else if (codeText !== undefined) {
      result.push(
        <code
          key={`c-${key++}`}
          className="rounded bg-paper px-1 py-0.5 font-mono text-[13px] text-ink border border-ink-border"
        >
          {codeText}
        </code>
      );
    } else if (linkText !== undefined && linkUrl !== undefined) {
      const safeTarget = getSafeLinkTarget(linkUrl);
      if (!safeTarget) {
        result.push(linkText);
        lastIndex = regex.lastIndex;
        continue;
      }
      result.push(
        <a
          key={`l-${key++}`}
          href={safeTarget.href}
          target={safeTarget.isExternal ? "_blank" : undefined}
          rel={safeTarget.isExternal ? "noopener noreferrer" : undefined}
          className="text-cobalt hover:underline transition-colors focus:outline-none focus:ring-1 focus:ring-primary inline-flex items-center gap-0.5"
        >
          {linkText}
        </a>
      );
    } else if (autoUrl !== undefined) {
      result.push(
        <a
          key={`a-${key++}`}
          href={autoUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-cobalt hover:underline transition-colors focus:outline-none focus:ring-1 focus:ring-primary break-all"
        >
          {autoUrl}
        </a>
      );
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    result.push(text.slice(lastIndex));
  }

  return result;
}

/**
 * Parses raw table rows into an array of string cells.
 */
function parseTableRow(rowText: string): string[] {
  const trimmed = rowText.trim();
  let content = trimmed;
  if (content.startsWith("|")) content = content.slice(1);
  if (content.endsWith("|")) content = content.slice(0, -1);
  return content.split("|").map((cell) => cell.trim());
}

/**
 * Checks if a line is a markdown table separator row (e.g. |---|:---:|---:|).
 */
function isSeparatorRow(line: string): boolean {
  const trimmed = line.trim();
  return (
    /^[|\s:-]+$/.test(trimmed) &&
    trimmed.includes("-") &&
    trimmed.includes("|")
  );
}

/**
 * Checks if a line is a list item.
 */
function isListItem(line: string): boolean {
  return /^\s*[-*+]\s+/.test(line) || /^\s*\d+\.\s+/.test(line);
}

/**
 * Parses raw markdown text into block-level elements.
 */
function parseMarkdown(markdown: string): Block[] {
  const blocks: Block[] = [];
  const lines = normalizeArchiveText(markdown).split(/\r?\n/);
  let startLine = 0;

  // Many archived markdown files include YAML front matter. The document page
  // already exposes source metadata in the header, so the reader should begin
  // at the article body instead of rendering archival fields as prose.
  if (lines[0]?.trim() === "---") {
    const endFrontMatter = lines.findIndex((line, index) => index > 0 && line.trim() === "---");
    if (endFrontMatter > 0) {
      startLine = endFrontMatter + 1;
    }
  }

  for (let i = startLine; i < lines.length; i++) {
    const line = lines[i];

    // 1. Fenced Code Blocks
    if (line.trim().startsWith("```")) {
      const language = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      blocks.push({
        type: "code",
        language,
        code: codeLines.join("\n"),
      });
      continue;
    }

    // 2. Blockquotes
    if (line.trim().startsWith(">")) {
      const quoteLines: string[] = [];
      while (
        i < lines.length &&
        (lines[i].trim().startsWith(">") ||
          (lines[i].trim() !== "" && quoteLines.length > 0))
      ) {
        let content = lines[i].trim();
        if (content.startsWith(">")) {
          content = content.slice(1);
          if (content.startsWith(" ")) {
            content = content.slice(1);
          }
        }
        quoteLines.push(content);
        i++;
      }
      i--; // Adjust for loop increment
      blocks.push({
        type: "blockquote",
        lines: quoteLines,
      });
      continue;
    }

    // 3. Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      });
      continue;
    }

    // 4. Horizontal Rules
    if (/^(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,}$/.test(line.trim())) {
      blocks.push({ type: "hr" });
      continue;
    }

    // 5. Tables
    if (
      line.includes("|") &&
      i + 1 < lines.length &&
      isSeparatorRow(lines[i + 1])
    ) {
      const headers = parseTableRow(line);
      const separatorCells = parseTableRow(lines[i + 1]);
      const alignments = separatorCells.map((cell) => {
        const left = cell.startsWith(":");
        const right = cell.endsWith(":");
        if (left && right) return "center";
        if (right) return "right";
        return "left";
      });

      const rows: string[][] = [];
      i += 2; // Skip header and separator rows
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        lines[i].includes("|")
      ) {
        rows.push(parseTableRow(lines[i]));
        i++;
      }
      i--; // Adjust for loop increment

      blocks.push({
        type: "table",
        headers,
        alignments,
        rows,
      });
      continue;
    }

    // 6. Lists (Group consecutive items together)
    if (isListItem(line)) {
      const items: ListItem[] = [];
      const isOrdered = /^\s*\d+\.\s+/.test(line);

      while (i < lines.length && isListItem(lines[i])) {
        const currLine = lines[i];
        const uMatch = currLine.match(/^\s*[-*+]\s+(.*)$/);
        const oMatch = currLine.match(/^\s*\d+\.\s+(.*)$/);

        let itemText = "";
        if (uMatch) {
          itemText = uMatch[1];
        } else if (oMatch) {
          itemText = oMatch[1];
        }

        // Check for task list checkboxes: [ ] or [x]
        let checked: boolean | undefined = undefined;
        if (itemText.startsWith("[ ] ")) {
          checked = false;
          itemText = itemText.slice(4);
        } else if (itemText.startsWith("[x] ") || itemText.startsWith("[X] ")) {
          checked = true;
          itemText = itemText.slice(4);
        }

        items.push({ text: itemText, checked });
        i++;
      }
      i--; // Adjust for loop increment

      blocks.push({
        type: "list",
        ordered: isOrdered,
        items,
      });
      continue;
    }

    // 7. Paragraphs
    if (line.trim() !== "") {
      const paragraphLines: string[] = [line.trimEnd()];
      i++;
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !lines[i].trim().startsWith("```") &&
        !lines[i].trim().startsWith(">") &&
        !/^(#{1,6})\s+/.test(lines[i]) &&
        !/^(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,}$/.test(lines[i].trim()) &&
        !isListItem(lines[i]) &&
        !(
          lines[i].includes("|") &&
          i + 1 < lines.length &&
          isSeparatorRow(lines[i + 1])
        )
      ) {
        paragraphLines.push(lines[i].trimEnd());
        i++;
      }
      i--; // Adjust for loop increment

      blocks.push({
        type: "paragraph",
        text: paragraphLines.map((paragraphLine) => paragraphLine.trim()).join(" "),
        lines: paragraphLines,
      });
    }
  }

  return blocks;
}

export default function MarkdownReader({ content, className }: MarkdownReaderProps) {
  const blocks = React.useMemo(() => parseMarkdown(content), [content]);

  return (
    <div className={`prose-reader max-w-[72ch] select-text break-words ${className || ""}`}>
      {blocks.map((block, index) => {
        switch (block.type) {
          case "heading": {
            const level = block.level || 1;
            const text = block.text || "";
            if (level === 1) {
              return (
                <h2
                  key={index}
                  className="text-2xl font-bold tracking-tight text-ink mt-8 mb-4 border-b border-ink-border pb-2"
                >
                  {parseInline(text)}
                </h2>
              );
            }
            if (level === 2) {
              return (
                <h3
                  key={index}
                  className="text-xl font-bold tracking-tight text-ink mt-7 mb-3"
                >
                  {parseInline(text)}
                </h3>
              );
            }
            if (level === 3) {
              return (
                <h4
                  key={index}
                  className="text-lg font-semibold text-ink mt-6 mb-2"
                >
                  {parseInline(text)}
                </h4>
              );
            }
            if (level === 4) {
              return (
                <h5
                  key={index}
                  className="text-base font-semibold text-ink mt-5 mb-2"
                >
                  {parseInline(text)}
                </h5>
              );
            }
            return (
              <h6
                key={index}
                className="text-sm font-semibold uppercase tracking-wider text-ink-muted mt-4 mb-2"
              >
                {parseInline(text)}
              </h6>
            );
          }

          case "paragraph": {
            const lines = block.lines || [];
            const hasHardBreaks = lines.some((line) => /\s{2,}$/.test(line));

            if (hasHardBreaks) {
              return (
                <p
                  key={index}
                  className="text-base leading-relaxed text-ink mb-4"
                >
                  {lines.map((line, lineIndex) => (
                    <React.Fragment key={lineIndex}>
                      {parseInline(line.replace(/\s{2,}$/, "").trim())}
                      {lineIndex < lines.length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </p>
              );
            }

            return (
              <p
                key={index}
                className="text-base leading-relaxed text-ink mb-4"
              >
                {parseInline(block.text || "")}
              </p>
            );
          }

          case "blockquote": {
            const lines = block.lines || [];
            return (
              <blockquote
                key={index}
                className="rounded border border-ink-border bg-paper px-4 py-3 my-5 text-ink italic leading-relaxed text-base"
              >
                {lines.map((line, idx) => (
                  <span key={idx} className="block min-h-[1.5rem]">
                    {parseInline(line)}
                  </span>
                ))}
              </blockquote>
            );
          }

          case "code": {
            const language = block.language || "";
            const code = block.code || "";
            return (
              <div
                key={index}
                className="my-5 overflow-hidden rounded border border-ink-border bg-paper font-mono text-[13px] leading-relaxed"
              >
                {language && (
                  <div className="flex items-center justify-between border-b border-ink-border bg-paper/60 px-4 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-muted">
                    <span>{language}</span>
                  </div>
                )}
                <pre className="overflow-x-auto p-4 text-ink select-text">
                  <code>{code}</code>
                </pre>
              </div>
            );
          }

          case "list": {
            const items = block.items || [];
            const isOrdered = block.ordered || false;
            const hasTaskItems = items.some((item) => item.checked !== undefined);

            if (hasTaskItems) {
              return (
                <ul key={index} className="list-none pl-1 my-4 space-y-2 text-base text-ink leading-relaxed">
                  {items.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2.5">
                      {item.checked !== undefined ? (
                        item.checked ? (
                          <span className="mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded border border-emerald-300 bg-emerald-50 text-emerald-700">
                            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          </span>
                        ) : (
                          <span className="mt-1 h-4 w-4 shrink-0 rounded border border-ink-border bg-pure" />
                        )
                      ) : (
                        <span className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-muted" />
                      )}
                      <span className="flex-1">{parseInline(item.text)}</span>
                    </li>
                  ))}
                </ul>
              );
            }

            if (isOrdered) {
              return (
                <ol key={index} className="list-decimal pl-6 my-4 space-y-2 text-base text-ink leading-relaxed">
                  {items.map((item, idx) => (
                    <li key={idx} className="pl-1">
                      {parseInline(item.text)}
                    </li>
                  ))}
                </ol>
              );
            }

            return (
              <ul key={index} className="list-disc pl-6 my-4 space-y-2 text-base text-ink leading-relaxed">
                {items.map((item, idx) => (
                  <li key={idx} className="pl-1">
                    {parseInline(item.text)}
                  </li>
                ))}
              </ul>
            );
          }

          case "table": {
            const headers = block.headers || [];
            const alignments = block.alignments || [];
            const rows = block.rows || [];

            return (
              <div key={index} className="my-6 overflow-x-auto rounded border border-ink-border bg-pure">
                <table className="w-full border-collapse text-left text-sm leading-normal">
                  <thead>
                    <tr className="border-b border-ink-border bg-paper text-xs font-bold uppercase tracking-wider text-ink-muted">
                      {headers.map((header, idx) => (
                        <th
                          key={idx}
                          className="px-4 py-2.5 font-semibold"
                          style={{ textAlign: alignments[idx] || "left" }}
                        >
                          {parseInline(header)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink-border text-ink">
                    {rows.map((row, rIdx) => (
                      <tr key={rIdx} className="hover:bg-paper/40 transition-colors">
                        {row.map((cell, cIdx) => (
                          <td
                            key={cIdx}
                            className="px-4 py-2.5 text-base"
                            style={{ textAlign: alignments[cIdx] || "left" }}
                          >
                            {parseInline(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }

          case "hr": {
            return <hr key={index} className="my-8 border-t border-ink-border" />;
          }

          default:
            return null;
        }
      })}
    </div>
  );
}
