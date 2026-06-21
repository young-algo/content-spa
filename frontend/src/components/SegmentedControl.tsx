import { cn } from "../lib/utils";

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}

/**
 * Shared segmented toggle (sort / mode / filter). The single source of truth
 * for the active cobalt-tint style — previously inlined as `filterBtn` in
 * Library, Search, and Ask.
 */
export default function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className,
}: SegmentedControlProps<T>) {
  return (
    <div className={cn("flex rounded-md border border-ink-border bg-paper p-0.5", className)}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            "rounded px-3 py-1.5 text-xs font-medium transition-colors focus:outline-none",
            value === opt.value
              ? "bg-cobalt-light/60 text-cobalt font-semibold"
              : "text-ink-muted hover:text-ink",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
