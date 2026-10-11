# AI Research Agent

Belajar fundamental agent: goal → LLM decision → tool call → observation →
state → loop → stop. Bukan platform riset production.

## Cara kerja

```
goal → Planner (pecah jadi 2-4 subtopik)
     → Researcher × N subtopik (search_web / read_page, loop sampai "final")
     → merge jawaban tiap subtopik
     → Critic (cek jawaban gabungan sudah nutup goal atau belum)
     → kalau "needs_more": 1 ronde riset tambahan buat gap itu, lalu Critic lagi
     → catat goal + jawaban ke memory.jsonl
     → report.json + report.md
```

Pipeline dibatasi maksimal 1 ronde riset tambahan (nggak di-loop sampai Critic
bilang "ok" terus-terusan) dan maksimal 3 subtopik — biar nggak gampang kena
rate limit provider LLM free tier.

Tiap Researcher jalanin loop eksplisit yang sama: LLM balas satu action JSON
(`search_web`, `read_page`, atau `final`) per giliran, hasil tool di-feed
balik sebagai observation, ulang sampai `final` atau limit iterasi/tool call
kena (lalu dipaksa rangkum).

Tool calling pakai prompt biasa (bukan `tools` param OpenAI-style) karena
model gratisan yang dipakai selama development hang/balas kosong kalau
`tools` disertakan — endpoint completion polos + instruksi format JSON di
system prompt jauh lebih stabil.

Memory (`memory.py`) nyimpen tiap run ke `memory.jsonl` dan nyari run lama
yang mirip lewat keyword overlap (bukan embedding) — kalau ketemu, jawaban
lama itu di-suntik sebagai konteks ke subtopik pertama.

## MCP

Selain 2 tool native (`search_web`, `read_page`), agent bisa jadi **MCP client**:
tool dari server MCP ditemukan otomatis lewat `list_tools` lalu masuk ke
registry yang sama. Analogi: tool native itu alat yang udah nempel di meja,
MCP itu stop kontak — colok server baru, alat barunya langsung kebaca tanpa
ngoding ulang.

Server didaftarin di `mcp_servers.json` (default: `mcp-server-fetch` via `uvx`,
tanpa API key):

```json
{ "fetch": { "command": "uvx", "args": ["mcp-server-fetch"] } }
```

- Tool tampil sebagai `<server>__<tool>` (mis. `fetch__fetch`) biar nggak
  pernah tabrakan sama tool native atau server lain.
- Argumen divalidasi dulu pakai JSON schema dari server (tipe `string`,
  `integer`, `boolean`, dst.) sebelum dikirim — salah tipe balik sebagai
  error, nggak sampai ke server.
- File `mcp_servers.json` dihapus / `{}` → agent jalan persis seperti tanpa MCP.
- Server gagal start (mis. `uvx` belum ada) → dicatat `[MCP] ... failed to
  start, skipping`, server lain dan tool native tetap jalan.
- Run **pertama** `mcp-server-fetch` bisa nge-log banyak `Failed to parse
  JSONRPC message` — itu server-nya nge-install dependensi dulu di stdout.
  Run berikutnya bersih.

## Setup

```
pip install -r requirements.txt
cp .env.example .env
```

`uvx` (dari [uv](https://docs.astral.sh/uv/)) dibutuhkan buat jalanin server
MCP default.

Isi `.env`:
- `TOKENROUTER_API_KEY` / `TOKENROUTER_BASE_URL` / `TOKENROUTER_MODEL` — LLM.
  Bisa TokenRouter (`z-ai/glm-5.3-free`) atau endpoint OpenAI-compatible lain
  (mis. Gemini) selama formatnya sama.
- `TAVILY_API_KEY` — buat `search_web` (tavily.com, free tier 1.000
  credits/bulan, no card).

## Jalanin

```
python agent.py "<goal riset>"
```

Output: `report.json` (data lengkap) + `report.md` (ringkasan baca).
Log progres (`[AGENT]`, `[PLANNER]`, `[RESEARCHER]`, `[CRITIC]`, `[MEMORY]`,
`[TOOL]`) ke stderr.

## Test

```
python test_agent.py
python test_memory.py
python test_mcp_client.py
```

Semua test pakai fake LLM / temp file — nggak ada network call, nggak butuh
API key. `test_mcp_client.py` nge-spawn server MCP palsu (file itu sendiri,
`--server`) lewat stdio beneran, jadi protokolnya ikut kebukti.
