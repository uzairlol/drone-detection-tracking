# data/raw/_credentials — credentials live here, never in git

Every download backend that needs authentication looks in this directory first.
The files are gitignored (`.baidu_cookies.json`, `kaggle.json`, `*_cookie.json`,
`*_token.json`). The `.example` files next to them are **not** gitignored and
contain no secrets — they are shape templates.

## Which backend needs what

| backend | datasets | credential | where it looks |
| --- | --- | --- | --- |
| `kaggle` | `dvb`, `mavvid` (fallback) | Kaggle API token | `~/.kaggle/kaggle.json`, `~/.kaggle/access_token` |
| `gdrive` | `antiuav` 300 | **none** | — |
| `modelscope` | `antiuav` 600 | **none** | — |
| `bitbucket` | `mavvid` | **none** | — |
| `baidu` | `antiuav` 410, `mmuav` | login cookie | `data/raw/_credentials/.baidu_cookies.json` |
| `http` | `dvb` annotations | **none** | — |

Good news: three of the four datasets need no account at all. Only `dvb` (Kaggle)
and the two Baidu-hosted variants need credentials.

## Kaggle — needed for `dvb`, the important one

```powershell
kaggle login --accept-dictionary
```

That writes `~/.kaggle/kaggle.json` **outside** this directory, and it is the
step most likely to be missing on a fresh office machine. The backend checks
`access_token`, `access_token.txt` *and* `kaggle.json`, because the modern CLI
writes the first and will happily ignore a stale `kaggle.json` that is present but
no longer valid — gating on `kaggle.json` alone reports "configured" when it is
not.

See `kaggle.json.example`. Verify with:

```powershell
anti-uav download --dataset dvb --dry-run
```

`--dry-run` prints the plan *and* whether credentials resolve, and touches
nothing. Run it before every large download.

## Baidu — needed for `mmuav`, the one that cannot be automated

Baidu requires a logged-in session. There is no API path. The workflow:

1. Log into pan.baidu.com in a normal browser.
2. Open the share link for your dataset. Links and extraction codes are printed by
   `anti-uav download --dataset mmuav --dry-run`, and recorded in
   `configs/datasets/registry.yaml`.
3. Solve the captcha and start the transfer in the official client. Let it run —
   throttling means this takes a while and may need restarting.
4. Export the session cookies to `.baidu_cookies.json` in **this** directory. See
   `.baidu_cookies.json.example` for the exact shape.

Then re-run the pipeline with `--skip-download` so the tooling builds its frame
index from what already landed rather than trying to fetch it again.

If you skip this, `anti-uav build --combo mmuav` **skips that source with a
warning** and the rest of the matrix still runs. It will not pretend mmuav was
there — check `build_report.json` for `skipped: true` before trusting a combo.

## If a secret ever gets committed

Rotate it at the publisher, then `git rm --cached` the file. Kaggle tokens and
Baidu session cookies are both account-wide; treat them as compromised.