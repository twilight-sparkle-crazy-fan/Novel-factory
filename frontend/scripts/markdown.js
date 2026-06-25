function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInline(value) {
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  text = text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return text;
}

function renderTextBlock(text) {
  const lines = text.replaceAll("\r\n", "\n").split("\n");
  const html = [];
  let listType = null;
  let paragraph = [];

  const closeParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) html.push(`</${listType}>`);
    listType = null;
  };

  for (const line of lines) {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const quote = line.match(/^>\s?(.*)$/);
    if (heading) {
      closeParagraph();
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
    } else if (unordered || ordered) {
      closeParagraph();
      const target = unordered ? "ul" : "ol";
      if (listType !== target) {
        closeList();
        listType = target;
        html.push(`<${listType}>`);
      }
      html.push(`<li>${renderInline((unordered || ordered)[1])}</li>`);
    } else if (quote) {
      closeParagraph();
      closeList();
      html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
    } else if (!line.trim()) {
      closeParagraph();
      closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  closeParagraph();
  closeList();
  return html.join("");
}

export function renderMarkdown(source) {
  const text = String(source || "");
  const blocks = [];
  let cursor = 0;
  const pattern = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) blocks.push(renderTextBlock(text.slice(cursor, match.index)));
    const language = escapeHtml(match[1].trim() || "代码");
    const code = escapeHtml(match[2].replace(/\n$/, ""));
    blocks.push(
      `<div class="code-block"><div class="code-block-header"><span>${language}</span>` +
        `<button type="button" class="copy-code">复制</button></div><pre><code>${code}</code></pre></div>`,
    );
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) blocks.push(renderTextBlock(text.slice(cursor)));
  return blocks.join("") || "";
}

export function escapeText(value) {
  return escapeHtml(value);
}
