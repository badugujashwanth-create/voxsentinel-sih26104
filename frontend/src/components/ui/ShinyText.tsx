import "./ShinyText.css";

interface ShinyTextProps {
  text: string;
  className?: string;
}

/** Applies a restrained, one-pass brand sheen without adding runtime state. */
export function ShinyText({ text, className = "" }: ShinyTextProps) {
  return (
    <span className={`shiny-text ${className}`.trim()} aria-label={text}>
      {text}
    </span>
  );
}
