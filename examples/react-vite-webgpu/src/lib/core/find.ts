import globToRegExp from "glob-to-regexp";

/**
 * Finds files that match a given glob pattern from a list of file paths.
 *
 * @example
 * ```ts
 * const result = findFiles({
 *   glob: "*.ts",
 *   filePaths: [
 *     "src/index.ts",
 *     "src/utils.js",
 *     "src/components/Component.tsx",
 *     "README.md"
 *   ]
 * });
 * console.log(result);
 * // Output:
 * // {
 * //   matched: ["src/index.ts", "src/components/Component.tsx"],
 * //   unmatched: ["src/utils.js", "README.md"]
 * // }
 * ```
 */
export function findFiles(options: { glob: string; filePaths: string[] }): {
  matched: string[];
  unmatched: string[];
} {
  const { glob, filePaths } = options;
  const regex = globToRegExp(glob, { extended: true });

  const matched: string[] = [];
  const unmatched: string[] = [];

  for (const filePath of filePaths) {
    if (regex.test(filePath)) {
      matched.push(filePath);
    } else {
      unmatched.push(filePath);
    }
  }

  return { matched, unmatched };
}
