/** Which build this is: "browser" (Cloudflare Pages, Pyodide engine) or wm-serve. Set by --define. */
export const BROWSER = process.env.WM_TARGET === "browser"
/** The public site's address, for the model-server setup steps (OLLAMA_ORIGINS). */
export const SITE_ORIGIN = BROWSER ? location.origin : ""
