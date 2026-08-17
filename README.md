# YouTube 海外访谈内容自动化 Agent

支持两种入口：输入一条带英文字幕的 YouTube Podcast / Interview URL，或让 Trend Agent 从 YouTube Podcast Top 100 中自动发现本月候选。最终输出一至多个可人工审核的中文社交媒体轮播图 Content Package。当前版本不包含自动发布、Dashboard 或数据库。

## 当前能力

```text
YouTube Podcast Top 100（可选）
  -> 本月单集指标
  -> Rule 硬过滤与播放表现评分
  -> MiMo 内容适配精排
  -> Top 10 YouTube URL
  ->
YouTube URL
  -> yt-dlp 元数据
  -> YouTube 人工/自动英文字幕（带时间戳）
  -> OpenAI 结构化 Editorial + Caption
  -> Transcript 强制回源与时间覆盖校验
  -> 按选题下载连续视频片段
  -> 每个时间戳附近三帧质量择优
  -> 1080x1350 原人物画面 + 中文字幕
  -> source.json / content.json / caption.md / metadata.json
```

模型只负责提出选题、时间段、Slide 时间戳和中文编辑结果。`original_text` 不接受模型填写，而是由系统按时间戳从 Transcript 反查生成，因此每个 Topic、Slide、截图都能回溯。

## 环境要求

- Python 3.12+
- `uv`
- `yt-dlp`
- `ffmpeg` / `ffprobe`
- 一个可用的 MiMo API Key（默认 Editorial/Caption）

macOS 上建议使用带常见编解码器的 FFmpeg。图片文字由 Pillow 和系统中文字体渲染，不依赖 FFmpeg 的字幕滤镜。

## 安装

```bash
uv sync --extra dev
cp .env.example .env
```

默认使用小米 MiMo V2.5。在 `.env` 填入：

```dotenv
EDITORIAL_PROVIDER=mimo
MIMO_API_KEY=...
MIMO_MODEL=mimo-v2.5
MIMO_BASE_URL=https://api.xiaomimimo.com/v1
```

Editorial 请求会显式设置 `thinking.type=disabled`。这是因为该任务需要稳定输出可校验 JSON，
不需要让内部推理占用大量输出 Token；最终结果仍会经过 Pydantic 和 Transcript 双重校验。
如果首次结果违反时间线合同，系统最多执行一次有边界的纠错：只把该候选对应、不超过
10 分钟的 Transcript 摘录交回 MiMo，要求重新选择 30–180 秒片段并重写全部内容；第二次
仍不合格就明确失败，不会无限付费重试。

通过时间合同后还会执行一次来源限定核查。核查器只能重写中文选题、Slide 和 Caption，
不能改变 Source Segment、时间戳或英文锚点；任何原文未明确支持的日期、身份、影响范围或
背景事实都必须删除，来源身份一旦被修改会直接失败。

如需切回 OpenAI，可设置 `EDITORIAL_PROVIDER=openai`，并配置 `OPENAI_API_KEY` 与
`OPENAI_MODEL`。

手工 URL-to-package 不需要 `YOUTUBE_API_KEY`。自动发现命令 `youtube-content-trend`
需要在 Google Cloud 启用 YouTube Data API v3，并把 Key 写入 `.env`：

```dotenv
YOUTUBE_API_KEY=...
```

该 Key 只用于读取公开 Playlist、发布时间、时长、字幕标记和播放量，不用于登录或修改
YouTube 内容。

## 自动发现本月 Top 10

Trend Agent 将最新 YouTube Podcast Top 100 周榜作为节目种子，通过 YouTube Data API
读取本月单集，再执行两层筛选：

1. Rule 硬过滤：月份、20 分钟至 4 小时、公开播放量、英文字幕标记、非直播、非 Short / Clip / Trailer。
2. Rule 评分：绝对播放量、日均播放量、节目榜单名次和发布时间新鲜度。
3. MiMo 精排：中文受众吸引力、观点密度、故事线、画面适配、嘉宾认知度和内容风险。
4. 最终分：55% Rule + 45% AI；同一节目优先最多保留 2 条，避免榜单被单一频道占满。

AI 不生成或修改播放量，只评估标题、频道、简介和节目背景。默认先由规则从全部候选筛到
30 条，再交给 MiMo，避免把 Top 100 的完整字幕全部送入模型。

```bash
uv run youtube-content-trend \
  --month 2026-08 \
  --output-dir outputs/trends \
  --max-seeds 100 \
  --preselect 30 \
  --top-n 10
```

输出：

```text
outputs/trends/2026-08/
├── trend-report.json  # 种子快照、规则结果、AI 六维评分和最终排名
└── top10.md           # 可点击的 Top 10 URL 与中文入选理由
```

如果要把排名前 3 的视频直接送入现有长图流水线：

```bash
uv run youtube-content-trend \
  --month 2026-08 \
  --top-n 10 \
  --generate-top 3 \
  --max-topics 1
```

`--generate-top` 会额外获取字幕、调用 Editorial 并下载选题片段；默认值为 `0`，即只输出
候选榜单，不产生后续长图费用。

### 无网络、无付费 API 的 Trend 演示

```bash
uv run youtube-content-trend \
  --month 2026-08 \
  --top-n 3 \
  --preselect 3 \
  --max-seeds 3 \
  --seed-fixture fixtures/trend_seeds.json \
  --video-fixture fixtures/trend_videos.json \
  --ai-fixture fixtures/trend_ai.json
```

fixture 必须显式传入，报告会标记 `fixture` provider，不会伪装成真实实时榜单。

## 正式运行

```bash
uv run youtube-content-agent \
  'https://www.youtube.com/watch?v=VIDEO_ID' \
  --output-dir outputs/my-run \
  --work-dir work/youtube-content-agent \
  --max-topics 3
```

如果 YouTube 对匿名访问触发验证，可在 `.env` 显式设置浏览器名，例如：

```dotenv
YT_DLP_COOKIES_FROM_BROWSER=chrome
```

系统不会导出或记录 Cookie。

## 无模型 API Key 的明确演示模式

fixture 不会被隐式启用，必须主动传入：

```bash
uv run youtube-content-agent \
  'https://www.youtube.com/watch?v=VIDEO_ID' \
  --editorial-fixture fixtures/demo_editorial.json \
  --output-dir outputs/demo \
  --max-topics 1
```

输出 `metadata.json` 会把 provider 标成 `fixture:...`，避免把 Mock 当成生产模型结果。fixture 的时间范围和 Slide 时间戳仍必须通过真实 Transcript 回源校验。

## Content Package

```text
outputs/my-run/
├── manifest.json
├── transcript.json
└── 01-topic-slug/
    ├── source.json
    ├── content.json
    ├── caption.md
    ├── metadata.json
    ├── storyboard.jpg
    └── images/
        ├── 01.jpg
        ├── 02.jpg
        └── ...
```

- `source.json`：视频身份、连续 Source Segment、该范围内完整原文和原始字幕段。
- `content.json`：中文选题、Hook、Caption、6-10 个 Slide；每个 Slide 同时保存时间戳、附近英文原文、中文和图片路径。
- `metadata.json`：字幕来源、Editorial provider、图片数、是否完整。
- `storyboard.jpg`：默认主成品，宽度 1080；顶部大人物画面，下方按时间顺序堆叠其余分镜条和通栏黑色字幕带。每个字幕条固定同一字号且只显示一行，长句会按标点和实际像素宽度拆成多个连续字幕条，行尾标点会自动移除；内容较多时画布会自动增高，避免缩小字体。
- `manifest.json`：整次运行的视频与 Package 清单。

## Transcript 策略

主路径使用 YouTube 已有的人工英文字幕；没有时使用 YouTube 自动英文字幕。系统只请求精确的 `en` 轨道，优先下载 `json3`，避免自动字幕 VTT 的滚动文本重复，并保留秒级开始/结束时间。成功结果会缓存到 `work/youtube-content-agent/<video_id>/`，避免长访谈重复请求字幕，同时不污染用户交付目录。

无字幕时默认明确失败，不会悄悄换供应商。完整音频 ASR 下载链尚未纳入本 MVP；后续接入时应显式启用，并使用本机 MLX Whisper Large V3 Turbo，不做 Qwen 或云端的隐式降级。

## Editorial 与 Timestamp 约束

- 选题必须对应 20-240 秒连续片段。
- 每个选题 6-10 个 Slide。
- Slide 时间戳必须递增且位于 Source Segment 内。
- 每张 Slide 必须携带一段逐字复制的英文 `source_quote`；系统在 Transcript 中找到该
  锚点后，将时间戳纠正到真实字幕段起点，并只保存 Transcript 派生的 `original_text`。
- Source Segment 必须被 Transcript 实际覆盖。
- 每张 Slide 的英文原文由时间戳反查，时间差超过 3 秒直接失败。
- 中文允许自然压缩，但 Prompt 明确禁止增加原文没有的事实。

## Screenshot 策略

每个选题优先只下载所需连续片段（前后各留 3 秒），避免下载数小时完整视频。若 YouTube 的远程媒体地址拒绝 FFmpeg seek，系统会明确记录 fallback：用 yt-dlp 原生方式下载一次 480p 以内源视频到 `work/` 缓存，随后在本地精确裁切；同一视频的多个选题复用该缓存。每个 Slide 在目标时间附近尝试 `-0.35s / 0s / +0.35s` 三帧，以曝光、对比度和清晰度做轻量评分，选出较自然的一帧。系统同时输出两种视觉：`images/` 保留独立 4:5 图片用于追溯或轮播，`storyboard.jpg` 则把全部分镜合成为一张宽度 1080 的连续故事长图，作为默认发布成品。字幕统一字号、逐条单行显示；若固定字号下内容超过 1440 高度，成品会自动增高而不是缩小文字。

第一版尚未做人脸检测、说话人识别、切镜头检测或人物安全区重排，这些属于下一阶段视觉质量优化。

## 测试与检查

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

端到端测试会在本地临时生成一段视频，完整验证：Transcript -> Editorial fixture -> 强制回源 -> 视频截帧 -> 中文字幕图片 -> 四类 Content Package 文件。

## 当前边界

- 没有 MiMo/OpenAI Key 时只能用显式 fixture 验证全链路，不能评价真实选题质量。
- 自动发现依赖 YouTube Data API v3；官方榜单本身是美国周榜、按观看时长排名，不等于全球月度单集播放榜，因此系统把它作为种子池，再按本月单集公开数据重排。
- Podcast Playlist 的维护顺序由频道决定；当前每个节目默认读取前 25 个条目，极高频或未按新到旧维护的 Playlist 可能漏掉本月旧一些的单集。
- 部分 YouTube 视频可能因地区、年龄、登录或 Bot 验证无法匿名下载。
- 超长 Transcript 目前一次提交给模型；生产化前应增加章节化候选召回与二阶段筛选。
- 当前截图评分不理解人物身份，仍需要人工检查构图、表情和字幕是否挡脸。
- 中文忠实度目前依靠 Prompt 与人工终审；后续应增加独立的双语 entailment/引用核查器。

## 下一阶段

先用 10-20 条真实长访谈建立编辑验收集，量化“至少一个可轻改发布”的命中率，再优化 Trend 权重、Editorial 召回和视觉选帧。榜单需要积累每日播放量快照，才能从“当前日均播放”升级为真实的播放增速曲线。
