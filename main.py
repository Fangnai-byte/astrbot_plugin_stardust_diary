#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Memory - AstrBot 智能记忆插件
====================================
- 监听群消息：所有消息先进入短期缓存（保留一天）
- 疑似重要的消息由 LLM 判断，重要则摘要进长期记忆
- 群聊触发 LLM 请求时，检索本群相关长期记忆注入 system_prompt
- 按群隔离：群与群之间的记忆互不可见
- 按 bot 分库：每个 self_id 独立 memory_<self_id>.db，多开实例互不串记忆
"""
import asyncio
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType, PermissionType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.message.components import Plain
from astrbot.core.platform.message_type import MessageType

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:  # 兼容旧版本 AstrBot
    def get_astrbot_data_path() -> str:
        return os.path.realpath(os.path.join(os.getcwd(), "data"))

# ---- 检索优化辅助（繁简归一化 / 探询意图）----
try:
    from opencc import OpenCC

    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None


def _norm(text: str) -> str:
    """繁体转简体；opencc 不可用时原样返回。"""
    if not text or _T2S is None:
        return text or ""
    try:
        return _T2S.convert(text)
    except Exception:
        return text or ""


# 提问疑似在确认"对方是否记得自己 / 双方关系"时的探询词
_PROBE_RE = re.compile(
    r"记得|记住|忘记|忘了|忘掉|记忆|还记得|記得|記住|忘記|記憶|認識|认识|"
    r"想起|想我|我是谁|我是誰|我的事|知道我|了解我|喜欢|喜歡|讨厌|討厭|爱|愛|"
    r"每天|平时|平時|经常|經常|总(?:是|会)|称呼|叫我|我的名字|叫(?:我|什么)"
)


def _clip(text: str, limit: int) -> str:
    """按字符数截断文本，超出时加省略号（用于限制注入体积）。"""
    s = (text or "").strip().replace("\n", " ")
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + "…"


def _is_profile(content: str, user_name: str = "") -> bool:
    """是否画像行。只认正文开头的【画像】标记，避免要点记忆里提到「画像」被误判。"""
    return (content or "").lstrip().startswith("【画像】") or "旧档" in (user_name or "")


def _profile_identity(content: str, user_name: str) -> str:
    """画像行归一化身份（= 显示名）：优先取正文『【画像】X：』里的 X，否则退回 user_name。

    注意：这是**显示名**，会随昵称改而变；做分组请用 `_profile_group_key()`。
    """
    c = (content or "").lstrip()
    if c.startswith("【画像】"):
        rest = c[len("【画像】"):]
        for sep in ("：", ":"):
            if sep in rest:
                name = rest.split(sep, 1)[0].strip()
                if name:
                    return SmartMemory._profile_key(name)
    return SmartMemory._profile_key(user_name)


# 画像正文里的 QQ（分组锚点）：`QQ=123456` / `QQ：123456`
_PROFILE_QQ_RE = re.compile(r"QQ\s*[=:：]\s*(\d{5,12})")


def _profile_qq(content: str) -> str:
    """从画像正文里取 QQ（取不到返回空串）。"""
    m = _PROFILE_QQ_RE.search(content or "")
    return m.group(1) if m else ""


def _profile_group_key(content: str, user_name: str) -> str:
    """画像的**分组键**：优先 QQ，取不到才退回归一化昵称。

    为什么不用昵称：昵称会改（`旧昵称` → `新昵称`），
    只按昵称分组会把同一个人裂成好几行档案（线上实测有人裂成 7 行）。
    QQ 是不变的锚点。
    """
    qq = _profile_qq(content)
    if qq:
        return "qq:" + qq
    return "name:" + SmartMemory._profile_key(user_name)


def _resolve_qq(name: str, nick2qq: dict) -> str:
    """把 LLM 返回的昵称认回 QQ。

    顺序：昵称里自带的 `(123456)`（prompt 里喂给模型的就是 `昵称(QQ)`）→
    与缓冲里的昵称精确匹配 → 去掉尾部括号备注后再匹配 → 只有唯一子串候选时才认。
    认不出来就返回空串（那就退回按昵称分组，与旧行为一致）。
    """
    n = (name or "").strip()
    if not n or not nick2qq:
        return ""
    m = re.search(r"[（(]\s*(\d{5,12})\s*[)）]", n)
    if m:
        return m.group(1)
    if n in nick2qq:
        return str(nick2qq[n])
    cleaned = SmartMemory._profile_key(n)
    if cleaned in nick2qq:
        return str(nick2qq[cleaned])
    cands = {str(q) for k, q in nick2qq.items() if k and (cleaned in k or k in cleaned)}
    return cands.pop() if len(cands) == 1 else ""


# 消息在说「说话人自己」时才值得带上他的画像（省 token）：
#   「锐评一下我」「我是谁」「我的事」「你们觉得我呢」
# 注意：「锐评」单独出现（如群里那句「宁宁求锐评」）也是"在问我"——实测没这条规则时
# 它没被认成自指问题，于是画像只有 80 字、模型只能泛泛回一句"话太少了"。
_SELF_RE = re.compile(
    r"(?:锐评|点评|评价)"
    r"|(?:说说|讲讲|聊聊|了解|认识|记得|记住|想起|知道|感觉|觉得|看|猜|分析|形容|描述)"
    r"[一下下]{0,2}(?:我|咱|本人)"
    r"|(?:我|咱|本人)(?:呢|吗|嘛|的(?:事|名字|情况|画像|档案|习惯|喜好)|是(?:谁|什么)|平时|最近|怎么样|如何)"
    r"|(?:你|您|你们|大伙|各位|大家)(?:觉得|看|认为|说说|评价|认识)[一下]{0,2}(?:我|咱|本人)"
)


def _like_escape(s: str) -> str:
    """把 LIKE 里的通配符转义掉（配合 `ESCAPE '\\'`）。

    坑：昵称是用户可控的，而 SQLite 的 LIKE 里 `%`/`_` 是通配符。
    实测昵称传 `%` 时 `user_name LIKE '%%%'` 会匹配**所有**画像行，
    于是别人的档案会被当成"说话人本人"注入（`feng_king` 这种带下划线的昵称一样中招）。
    """
    return (s or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _is_temp_recall(content: str, user_name: str = "") -> bool:
    """时效性内容（有人设保质期的「旧档」画像）不参与普通关键词召回。"""
    return "旧档" in (user_name or "")


def _sorted_pairs(d):
    """稳定排序（键升序），让合并结果在多次迁移间可复现。"""
    return sorted(d.items(), key=lambda kv: kv[0])


# 同义键归一化：历史画像里同一件事被写成很多近义键，合并后会各留一份导致重复。
# 收录两类规则：
#   1) 纯同义 → 直接归并（如 昵称/称呼对象 → 称呼），不会丢信息；
#   2) 语义有重叠但不完全等价 → 改名为独立键（如 身份/关注 → 关注领域），
#      既让同一件事不再各占一份，又不丢旧表述。
_PROFILE_KEY_ALIASES = {
    "qq/id": "QQ", "qq号": "QQ", "用户id": "QQ", "id": "QQ", "另一个号": "QQ",
    "昵称": "称呼", "称呼对象": "称呼", "称呼对方": "称呼", "被称呼": "称呼",
    "身份/自称": "身份", "身份线索": "身份补充",
    "身份/关注": "关注领域", "身份/活动": "活动",
    "口头禅/用语": "口头禅", "常用语气": "口头禅", "常用语": "口头禅",
    "常用工具": "工具",
    "常用路径": "路径",
    "常用命令": "命令",
    "github仓库": "重要仓库", "相关仓库": "重要仓库", "github相关": "重要仓库",
    "喜欢/偏好": "喜欢", "喜欢/在意": "喜欢",
    "被评价": "评价",
}

# 「与<某人>关系」→「关系」、「<某人>反应」进高优先档，取决于 _PROFILE_NAMES 里有哪些名字。
# 名字来源：配置项 profile_names（手填，例如对方的名字）+ **自动识别出来的各 bot 名字**
# （AstrBot 平台实例名，或用 bot_names 配置显式指定）。代码里不写死任何角色名。
_PROFILE_NAMES: tuple[str, ...] = ()


def _split_names(raw) -> list[str]:
    """把「逗号/顿号/分号/空格分隔的字符串」或列表拆成名字列表（保序去重）。"""
    if isinstance(raw, str):
        items = re.split(r"[,，、;；\s]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x) for x in raw]
    else:
        items = []
    out: list[str] = []
    for x in items:
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def set_profile_names(raw) -> None:
    """设置「画像别名名字」（覆盖用）。见 _canon_key / _profile_key_priority。"""
    global _PROFILE_NAMES
    _PROFILE_NAMES = tuple(_split_names(raw))


def add_profile_names(extra) -> None:
    """追加「画像别名名字」（自动识别出来的 bot 名字走这里，不覆盖手填的配置）。"""
    global _PROFILE_NAMES
    merged = list(_PROFILE_NAMES)
    for x in _split_names(extra):
        if x not in merged:
            merged.append(x)
    _PROFILE_NAMES = tuple(merged)


def _canon_key(key: str) -> str:
    """把同义画像键收敛到统一写法。"""
    k = (key or "").strip()
    if not k:
        return k
    canon = _PROFILE_KEY_ALIASES.get(k.lower(), k)
    low = canon.lower()
    for name in _PROFILE_NAMES:
        if low == f"与{name}关系".lower():
            return "关系"
    return canon


# 纯语气词 / 寒暄：这类消息本身不含信息，拿它去检索长期记忆只会捞出无关旧事
_QUERY_FILLER_CHARS = set("好~～唔嗯啊哦噢诶欸嘿哈呀吧呢吗嘛的了么哼嘻噗呜哇额呃哈姆喵惹哦哦呀哎呀")
_QUERY_FILLER_WORDS = {
    "在吗", "在么", "在不在", "在嘛", "你好", "早上好", "午好", "中午好", "晚上好",
    "晚安", "安", "安~", "早", "早呀", "嗨", "hi", "hello", "ok", "okk", "好的",
    "好的呢", "好哦", "嘿嘿", "哈哈", "嗯嗯", "谢谢", "谢啦", "收到", "okay",
}


def _is_weak_query(query: str) -> bool:
    """判断一条消息是否"纯语气/寒暄"——不含可检索的实质内容。"""
    q = (query or "").strip()
    if not q:
        return True
    compact = re.sub(r"[\s，。、！？!?.,~～…·\-—_]+", "", q).lower()
    if not compact:
        return True
    if compact in _QUERY_FILLER_WORDS:
        return True
    # 短且字符几乎全是语气词/口头禅（如「好~」「唔....」「欸嘿~」）
    if len(compact) <= 2 and all(ch in _QUERY_FILLER_CHARS for ch in compact):
        return True
    if len(compact) <= 4 and sum(ch in _QUERY_FILLER_CHARS for ch in compact) >= len(compact) - 1:
        return True
    # 语气词打头 + 很短（如「唔...所以」「唔....」）：仍属寒暄，不含可检索内容
    filler_n = sum(ch in _QUERY_FILLER_CHARS for ch in compact)
    if compact[0] in _QUERY_FILLER_CHARS and len(compact) <= 5:
        return True
    if len(compact) <= 8 and filler_n / len(compact) >= 0.5:
        return True
    return False


# 画像键优先级：数字越大越优先保留。默认 50，没登记的键按 50 处理。
# 目的：注入有字数上限时，别按字母序瞎丢——把"他是谁/你俩什么关系"留下，
# 把项目版本号、预算这类低价值键挤出去。
PROFILE_KEY_PRIORITY = {
    # 90：核心身份与关系，永远优先（QQ 是画像分组锚点，丢了会导致同人裂成多行）
    90: ("身份", "身份补充", "关系", "称呼", "擅长", "语言", "身体", "昵称", "QQ"),
    # 70：稳定偏好与互动方式
    70: ("喜欢", "互动", "互动方式", "习惯", "习惯动作", "口头禅", "饮食", "在意",
         "关心", "近期情绪", "近期感受", "近期个人事件", "重要态度", "遗憾", "事件",
         "共同记忆", "互动事件", "亲密互动方式", "问候习惯", "日常", "特征",
         "生理反应", "手", "家", "技能", "早餐", "评价", "行为线索", "被提醒",
         "备注"),
    # 50：默认档（未登记键）
    # 20：低价值元信息，最容易先丢
    20: ("项目", "项目版本", "预算", "挑战时长", "许可证决策", "许可偏好", "需求重点",
         "设备", "设备注意", "设备限制", "相关图片", "关注领域", "活动",
         "重要仓库", "路径", "命令", "工具", "语言习惯", "来源", "别名", "署名", "风格"),
}

# 有时效的「动态」键：超出 profile_dynamic_ttl_days 天就不再注入（只影响注入，不删库）
PROFILE_DYNAMIC_KEYS = (
    "近期情绪", "近期感受", "近期个人事件", "事件", "预算", "遗憾", "被提醒", "行为线索",
)


def _profile_key_priority(key: str) -> int:
    k = (key or "").strip()
    for name in _PROFILE_NAMES:
        if k == f"{name}反应":
            return 70
    for pri, keys in PROFILE_KEY_PRIORITY.items():
        if k in keys:
            return pri
    return 50


def _profile_drop_dynamic(
    d: dict, created_at: float | None, ttl_days: int, kv_ts: dict | None = None
) -> tuple[dict, list[str]]:
    """按 TTL 去掉过期的动态键。返回 (保留的字典, 被去掉的键名列表)。

    ★ 时效锚点用**每个键自己的时间戳**（`kv_ts`，见 `long_term.kv_ts` 列）：
    以前只看整行的 created_at，而合并时它会被刷成 now，于是只要这个人还在被整理，
    「近期情绪」这类动态键**永远不会过期**（值可能是几周前的还当"近期"注入）。
    `kv_ts` 里没有的键（老数据）退回用 created_at，行为与旧版一致。
    ttl_days <= 0 表示不做时效过滤。
    """
    if not ttl_days or ttl_days <= 0:
        return d, []
    kv_ts = kv_ts or {}
    now = time.time()
    kept, dropped = {}, []
    for k, v in d.items():
        if k not in PROFILE_DYNAMIC_KEYS:
            kept[k] = v
            continue
        ts = kv_ts.get(k)
        try:
            ts = float(ts) if ts else (float(created_at) if created_at else None)
        except (TypeError, ValueError):
            ts = float(created_at) if created_at else None
        if ts and (now - ts) / 86400.0 > ttl_days:
            dropped.append(k)
        else:
            kept[k] = v
    # 注意：这里**不做** `kept or d` 的兜底。以前"全被丢光就整条留下"，
    # 于是只剩一条过期动态键的画像（例如只有「近期情绪=三周前」）会被整条注入，
    # 反而把当轮语境带偏。调用方（_profile_for_inject）会在结果为空时返回空串，
    # 由 on_llm_req 跳过——宁可这一轮不带画像，也别塞一条过期情绪。
    return kept, dropped


def _kv_ts_load(raw) -> dict:
    """把 `kv_ts` 列的 JSON 解析成 {键: 时间戳}；坏数据当空。"""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return {str(k): v for k, v in d.items()} if isinstance(d, dict) else {}
    except Exception:
        return {}


def _kv_ts_dump(d: dict) -> str:
    """序列化 {键: 时间戳}；只留能转成数字的值，避免脏数据进库。"""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return json.dumps(out, ensure_ascii=False) if out else ""


def _pack_profile(
    identity: str,
    d: dict,
    max_keys: int,
    max_chars: int,
    created_at: float | None = None,
    ttl_days: int = 0,
    kv_ts: dict | None = None,
) -> str:
    """把键值字典打包成一行画像。

    超限时的取舍顺序（重要）：先按 TTL 去掉过期动态键 → 按优先级排序
    （同优先级按键名）→ 超字数时从**最低优先级**开始丢，而不是按字母序。
    """
    work = dict(d or {})
    if ttl_days:
        work, _dropped = _profile_drop_dynamic(work, created_at, ttl_days, kv_ts)
    items = sorted(work.items(), key=lambda kv: (-_profile_key_priority(kv[0]), kv[0]))
    if len(items) > max_keys:
        items = items[:max_keys]
    while True:
        content = "【画像】%s：%s" % (
            identity,
            "；".join("%s=%s" % (k, v) for k, v in items),
        )
        if len(content) <= max_chars or len(items) <= 1:
            return content
        # 丢当前优先级最低的那个（items 已按优先级降序，取最后一个）
        items = items[:-1]


# 命中这些词的消息才值得交给 LLM 判断（节省 token）
class SmartMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        set_profile_names(config.get("profile_names", ""))
        self._bot_names: dict[str, str] = {}   # self_id(QQ) -> 这个 bot 的名字（自动学到/配置指定）
        data_dir = get_astrbot_data_path()
        self.db_dir = os.path.join(data_dir, "smart_memory")
        os.makedirs(self.db_dir, exist_ok=True)
        self._legacy_db = os.path.join(self.db_dir, "memory.db")  # 旧版单库（升级后仅迁移一次）
        self._ensured: set = set()       # 已初始化（建表）的 bot 库
        self._organizing: set = set()    # 正在整理的群，避免并发重复触发（实例内）
        # ★ 插件被卸载/重载时由 terminate() 置真。整理要跑十几秒，旧实例的迟到结果
        #   若照写，而新实例也整理了同一批消息，同一批就成了两条记忆（10-11 实测）。
        self._stopped: bool = False
        self._consolidated: set = set()  # 已做过画像合并的 (库, 作用域)
        self._organize_backoff: dict = {}  # 群 -> 上次整理失败时间（失败退避用）
        self._last_purge: float = 0.0      # 上次物理清理过期短期记忆的时间
        self._refresh_profile_names()    # 把自动识别到的 bot 名字并进画像别名
        self._log(
            f"[{self._log_tag()}] bot 名字: "
            + ("、".join(self._known_bot_names()) or "（未识别，将随消息自动识别）")
            + "；画像别名名字: "
            + ("、".join(_PROFILE_NAMES) if _PROFILE_NAMES else "（无）")
        )
        # 把"省 token 的当前策略"落到日志里，出问题时一眼能看出配置有没有生效
        self._log(
            "[{0}] 缓冲策略: record_scope={1}、min_msg_chars={2}、禁用作用域={3}、"
            "单次整理上限={4}字；注入 top_k={5}、画像≤{6}字、本人画像注入={7}".format(
                self._log_tag(),
                self.config.get("record_scope", "all"),
                self.config.get("min_msg_chars", 0),
                self.config.get("disabled_scopes") or "（无）",
                self.config.get("organize_max_chars", 12000),
                self.config.get("top_k", 5),
                self.config.get("profile_max_chars", 120),
                self.config.get("profile_inject_mode", "self"),
            )
        )
        self._init_all_db()              # 为已存在的各 bot 库建表并清理过期

    async def terminate(self):
        """插件被卸载 / 重载时由框架调用（必须定义在本类里，框架用 __dict__ 判断）。

        为什么要它：整理一次要跑十几秒到一分钟，重载后**旧实例的协程还在跑**
        （实测 19:56:48 开始的那次整理在 19:57:02 重载之后，仍于 19:58:08 写了库），
        而新实例会把同一批消息再整理一遍 → 同一批消息两条记忆、还多烧一次上万字符的请求。
        这里只置一个标记：迟到的整理结果在写库前看到标记就作废，**消息留在缓冲里**
        下次照常整理，不丢记忆，只是延后。
        """
        self._stopped = True
        self._log(
            f"[{self._log_tag()}] 已停止：正在跑的整理不再写库（消息留在缓冲，下次重来）"
        )

    # ---------------- 数据库（按 bot self_id 分库） ----------------
    def _db_path(self, self_id=None) -> str:
        """每个 bot 独立库文件 memory_<self_id>.db，多开实例互不串记忆。"""
        sid = re.sub(r"[^0-9A-Za-z_-]", "_", str(self_id or "unknown"))
        return os.path.join(self.db_dir, f"memory_{sid}.db")

    def _connect(self, db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def _schema(cls, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT DEFAULT '',
                content TEXT NOT NULL,
                keywords TEXT DEFAULT '',
                raw TEXT DEFAULT '',
                created_at REAL NOT NULL,
                kv_ts TEXT DEFAULT ''
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_long_group ON long_term(group_id)")
        # 老库升级：补 kv_ts 列（每个画像键"最后一次被更新"的时间戳，JSON）
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(long_term)")}
            if "kv_ts" not in cols:
                conn.execute("ALTER TABLE long_term ADD COLUMN kv_ts TEXT DEFAULT ''")
                logger.info("[SmartMemory] 已为 long_term 补上 kv_ts 列（画像键级时效）")
        except Exception as e:
            logger.warning(f"[SmartMemory] 补 kv_ts 列失败（不影响其它功能）: {e}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS short_term (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT DEFAULT '',
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                expire_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_short_group ON short_term(group_id)")

    def _ensure_db(self, self_id=None) -> str:
        """确保某 bot 的库已建表；旧版 memory.db 只迁移一次给首个 bot。"""
        db_path = self._db_path(self_id)
        if db_path in self._ensured:
            return db_path
        if not os.path.exists(db_path) and os.path.exists(self._legacy_db):
            try:
                # 旧版单库 → 首个触发的 bot 的库；改名后其它 bot 不会再重复复制
                os.replace(self._legacy_db, db_path)
                logger.info(
                    f"[SmartMemory] 旧库已迁移: memory.db -> {os.path.basename(db_path)}"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"[SmartMemory] 旧库迁移失败: {e}")
        try:
            with closing(self._connect(db_path)) as conn, conn:
                self._schema(conn)
        except Exception as e:
            logger.warning(f"[SmartMemory] 初始化数据库失败 {db_path}: {e}")
        self._cleanup_expired(db_path)
        self._ensured.add(db_path)
        return db_path

    def _init_all_db(self):
        """启动时为所有已存在的 memory_*.db 建表、清理过期，并合并同一人的多行画像。"""
        try:
            for name in os.listdir(self.db_dir):
                if re.fullmatch(r"memory_[0-9A-Za-z_-]+\.db", name):
                    p = os.path.join(self.db_dir, name)
                    try:
                        with closing(self._connect(p)) as conn, conn:
                            self._schema(conn)
                        self._cleanup_expired(p)
                        # ★ 合并放这里：_ensure_db() 因为库里已登记会直接 return，
                        #   放在那边的扫描永远不执行（老 bug）。
                        self._consolidate_all_scopes(p)
                        self._ensured.add(p)
                    except Exception as e:
                        logger.warning(f"[SmartMemory] 初始化库 {name} 失败: {e}")
        except Exception as e:
            logger.warning(f"[SmartMemory] 扫描数据库目录失败: {e}")

    def _consolidate_all_scopes(self, db_path: str) -> int:
        """对某个库里的所有作用域做一次画像合并（幂等）。"""
        total = 0
        try:
            scopes = [
                row[0]
                for row in self._connect(db_path).execute(
                    "SELECT DISTINCT group_id FROM long_term"
                )
            ]
            logger.info(
                "[SmartMemory] 画像合并检查 %s：%d 个作用域",
                os.path.basename(db_path), len(scopes),
            )
            for scope in scopes:
                removed = self._consolidate_profiles(db_path, scope)
                if removed:
                    logger.info(
                        "[SmartMemory] 画像合并完成 %s：删除冗余 %d 行", scope, removed
                    )
                total += removed
        except Exception as e:
            logger.warning(f"[SmartMemory] 画像合并扫描失败: {e}")
        return total

    def _cleanup_expired(self, db_path: str):
        try:
            with closing(self._connect(db_path)) as conn, conn:
                conn.execute(
                    "DELETE FROM short_term WHERE expire_at < ?", (time.time(),)
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 清理过期短期记忆失败: {e}")

    # ---------------- 消息监听 ----------------
    def _record_filter_reason(self, event, text: str, group_id: str):
        """该不该把这条消息记进缓冲？返回 None=记，返回字符串=不记的原因。

        为什么要这道闸：**整理的开销正比于"进入缓冲的消息条数"**（阈值只是把
        「频次」和「单次体积」互换，不改变总量）。群里水消息极多（实测平均每条
        17 字，一半是「哈哈」「喵」这类），全记下来就等于按群活跃度烧 token。
        """
        scopes = self.config.get("disabled_scopes") or []
        if isinstance(scopes, str):
            scopes = [x.strip() for x in re.split(r"[,，;；\s]+", scopes) if x.strip()]
        if str(group_id) in {str(x).strip() for x in scopes}:
            return "该作用域在 disabled_scopes 里"

        # ★ 私聊不做「唤醒」和「长度」两道过滤：
        #   私聊每条都是对着 bot 说的（不存在"没叫它"），而且「嗯」「好」「晚安」
        #   这类短句恰恰是语境的一部分；水消息问题只存在于群聊。
        is_private = str(group_id).startswith("priv_")
        mode = str(self.config.get("record_scope", "all") or "all").strip().lower()
        is_bot = str(event.get_sender_id()) == str(event.get_self_id())
        if mode == "bot":
            if not is_bot:
                return "record_scope=bot（只记 bot 自己的发言）"
        elif is_bot:
            return "bot 自己的发言（避免自我循环）"
        elif mode == "triggered" and not is_private:
            # 「被唤醒/被回复到」的消息才记：开销随 bot 活跃度走，而不是随群活跃度
            try:
                woke = bool(getattr(event, "is_at_or_wake_command", False)) or bool(event.is_wake_up())
            except Exception:
                woke = True          # 判断不了就照旧记，别把记忆搞没了
            if not woke:
                return "record_scope=triggered（这条没唤醒 bot）"

        try:
            min_chars = int(self.config.get("min_msg_chars", 0) or 0)
        except (TypeError, ValueError):
            min_chars = 0
        if min_chars and not is_private and len(text) < min_chars:
            return f"短于 min_msg_chars={min_chars}"
        return None

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE
    )
    async def on_group_message(self, event: AstrMessageEvent):
        self_id = str(event.get_self_id())
        text = event.get_message_str().strip()
        if not text:
            return
        # 插件自身指令不参与记忆
        if text.startswith("/mem"):
            return
        group_id = self._get_scope(event)
        reason = self._record_filter_reason(event, text, group_id)
        if reason:
            logger.debug("[SmartMemory] 不入缓冲（%s）：%s", reason, text[:30])
            return
        user_id = str(event.get_sender_id())
        user_name = event.get_sender_name() or user_id

        # 1. 消息进短期缓冲（按 bot 分库）
        self._ensure_db(self_id)
        self._learn_bot_name(self_id, event)   # 顺手识别这个 bot 叫什么（UMO 前缀）
        self._add_short(self_id, group_id, user_id, user_name, text)

        # 2. 满阈值触发 AI 整理（异步，不阻塞回复）
        if self.config.get("organize_enabled", True):
            n = self._count_short(self_id, group_id)
            if n >= int(self.config.get("organize_threshold", 100)):
                asyncio.create_task(self._organize(self_id, group_id))

    # ---------------- 满阈值 AI 整理 ----------------
    def _organize_claims(self) -> dict:
        """跨实例的整理占位表，**挂在框架 context 上**（重载后新实例仍看得见旧实例）。

        本地 self._organizing 是实例内的锁，重载后新实例拿到的是新集合，挡不住旧实例：
        实测「重载撞上整理」时，同一个群的同一批消息被整理了两遍（多写 13/14 行记忆）。
        """
        try:
            reg = getattr(self.context, "_stardust_organize_claims", None)
            if not isinstance(reg, dict):
                reg = {}
                setattr(self.context, "_stardust_organize_claims", reg)
            return reg
        except Exception:
            return {}      # 拿不到就退化成"没有跨实例保护"，不影响单实例的正确性

    @staticmethod
    def _claim_is_mine(claims: dict, job_key: str, token: int) -> bool:
        v = claims.get(job_key)
        return isinstance(v, tuple) and len(v) == 2 and v[1] == token

    def _batch_alive(self, self_id: str, group_id: str, watermark: int) -> bool:
        """本批（直到水位线那一行为止）是否还在缓冲里。

        别的实例可能已经把这一批整理掉并删了缓冲行，此时再写一遍就是重复记忆。
        查不了就按"还在"处理——宁可多写一次，也别把该记的丢了。
        """
        if not watermark:
            return True    # 没有水位线时本来也不删缓冲（旧行为），这里不做判断
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                row = conn.execute(
                    "SELECT 1 FROM short_term WHERE id = ? AND group_id = ? LIMIT 1",
                    (int(watermark), group_id),
                ).fetchone()
            return row is not None
        except Exception:
            return True

    async def _organize(self, self_id: str, group_id: str):
        """攒满阈值后：AI 提炼人物画像键值+要点记忆，存长期后清空缓冲。"""
        # 键带上 self_id：两个 bot 同时挂在同一个群时，各自的整理不该互相挡
        job_key = f"{self_id}:{group_id}"
        if job_key in self._organizing:
            self._log(f"[{self._log_tag(self_id)}] 群 {group_id} 正在整理中，跳过本次触发")
            return
        # 失败退避：解析失败 / 拿不到供应商时，若不加退避，下一条消息就会再烧一次
        # 上万字符的请求（实测一次整理 15~45 秒），群活跃时等于按条计费地空转。
        last_fail = self._organize_backoff.get(job_key, 0.0)
        if last_fail and time.time() - last_fail < self.ORGANIZE_FAIL_BACKOFF:
            return
        self._organizing.add(job_key)
        claims = self._organize_claims()
        token = id(self)
        try:
            # ★ 跨实例占位：重载后旧实例可能还在跑同一批，直接放弃本次触发，
            #   省掉一次上万字符的请求（实测重复整理烧掉的正是这一份）。
            now = time.time()
            for k, v in list(claims.items()):
                ts = v[0] if isinstance(v, tuple) else v
                try:
                    stale = now - float(ts) > self.ORGANIZE_CLAIM_TTL
                except (TypeError, ValueError):
                    stale = True
                if stale:
                    claims.pop(k, None)   # 占位过期（上个实例可能已崩）→ 回收，别卡住这个群
            if job_key in claims and not self._claim_is_mine(claims, job_key, token):
                self._log(
                    f"[{self._log_tag(self_id)}] 群 {group_id} 有另一个实例正在整理，跳过本次触发"
                )
                return
            claims[job_key] = (now, token)
            self._log(f"[{self._log_tag(self_id)}] _organize 被调用，群 {group_id}")
            threshold = int(self.config.get("organize_threshold", 100))
            # 取数上限不能小于阈值，否则阈值被调大后就永远凑不满、永远不整理
            msgs = self._all_short(self_id, group_id, max(500, threshold))
            self._log(f"[{self._log_tag(self_id)}] 群 {group_id} 缓冲 {len(msgs)} 条")
            if len(msgs) < threshold:
                return
            # ★ 记下这批消息的最大 id：整理要跑十几秒，这期间新消息还在往缓冲里写，
            #   收尾时只能删到这条为止，否则会**静默丢掉这段时间的新消息**。
            watermark = self._short_watermark(msgs)
            try:
                max_chars = int(self.config.get("organize_max_chars", 12000) or 12000)
            except (TypeError, ValueError):
                max_chars = 12000
            # ★ 省 token（无损）：原来每条都写成「昵称(QQ): 内容」，实测 QQ 那截占了
            #   整个 prompt 的一半左右（缓冲区平均昵称 5.8 字 + QQ 10 字 = 18.8 字，
            #   而正文才 16.6 字）。现在改成开头放一张【群成员QQ对照】表，正文只留
            #   「昵称: 内容」。认人本来就用 buffer 里的 昵称→QQ 映射（_resolve_qq），
            #   不依赖 prompt 里逐条带 QQ；画像的 QQ 锚点也由 _merge_profile 自己写。
            cap = max(1000, max_chars)
            # ★ 只把「真的装进 prompt 的那些行」算作已处理：
            #   body 会被 cap 截断，若水位线仍取整批的最大 id，被截掉的消息就
            #   **从没进过模型却被删了**（调小 organize_max_chars 时特别容易踩）。
            #   这里逐行累加，记下最后一行装进去的 id 当水位线，剩下的留在缓冲里下轮再处理。
            picked, used, last_id = [], 0, 0
            for r in msgs:
                line = f"{(r['user_name'] or '').strip() or r['user_id']}: {r['content']}"
                if used + len(line) + 1 > cap - 400 and picked:   # 留 400 字给开头的对照表
                    break
                picked.append((r, line))
                used += len(line) + 1
                last_id = int(r["id"])
            speakers, seen_qq = [], set()
            for r, _line in picked:
                u = str(r["user_id"] or "")
                n = (r["user_name"] or "").strip() or u
                if u and u != "system" and u not in seen_qq:
                    seen_qq.add(u)
                    speakers.append((n, u))
            roster = "；".join(f"{n}={u}" for n, u in speakers[:80])
            text = ("【群成员QQ对照】" + roster + "\n\n"
                    + "\n".join(line for _r, line in picked))[:cap]
            if last_id:
                watermark = last_id
            else:                      # 一行都装不下就宁可不删
                watermark = 0
            self._log(
                f"[{self._log_tag(self_id)}] 本次整理喂入 {len(picked)}/{len(msgs)} 条"
                f"（{len(text)} 字），水位线 id={watermark}"
            )
            provider = self._pick_provider()
            self._log(f"[{self._log_tag(self_id)}] provider: {provider.meta().id if provider else None}")
            if provider is None:
                self._organize_backoff[job_key] = time.time()
                return
            resp = await provider.text_chat(
                prompt=text,
                system_prompt=(
                    "你是群聊档案管理员。根据聊天记录做两件事：\n"
                    "1. 为活跃成员建立人物画像，用键值对记录稳定属性（如 生日/喜欢/讨厌/身份/口头禅），"
                    "只记录明确出现过的信息，不要编造。\n"
                    "2. 提取值得长期记住的要点（约定、事件、计划、重要信息）。\n"
                    "记录格式：开头是一张【群成员QQ对照】表（昵称=QQ），之后每行是「昵称: 内容」。"
                    "写画像时请额外带上 QQ 键，值从对照表里查（例如 QQ=10001）。\n"
                    "只输出一个JSON对象："
                    '{"profiles": [{"user": "昵称", "attrs": {"键": "值"}}], '
                    '"memories": [{"content": "要点", "keywords": ["关键词"]}]}'
                    + (f"\n\n额外要求：{self.config.get('organize_prompt', '')}"
                       if self.config.get("organize_prompt", "") else "")
                ),
            )
            out = "".join(
                [c.text for c in (resp.result_chain.chain if resp.result_chain else []) if isinstance(c, Plain)]
            )
            self._log(f"[{self._log_tag(self_id)}] LLM 返回前100字: {out[:100]!r}")
            data = self._parse_json(out)
            self._log(f"[{self._log_tag(self_id)}] 解析结果: {bool(data)}")
            if not data:
                self._organize_backoff[job_key] = time.time()
                self._log(f"[{self._log_tag(self_id)}] 解析不出 JSON，{self.ORGANIZE_FAIL_BACKOFF}s 内不再重试本群")
                return
            # ★ 写库前的两道闸（10-11 实测的"同一批消息写两遍"）：
            #   ① 插件已被卸载/重载（terminate 置位）→ 旧实例的迟到结果作废；
            #   ② 这一批已被别的实例整理掉并从缓冲删了 → 再写就是重复。
            #   两种情况都**不写库也不删缓冲**：消息留在缓冲里，下次整理照常处理，不丢记忆。
            stopped = getattr(self, "_stopped", False)
            if stopped or not self._batch_alive(self_id, group_id, watermark):
                self._log(
                    f"[{self._log_tag(self_id)}] 群 {group_id} 整理结果作废"
                    f"（{'插件已停止' if stopped else '本批已被其它实例处理'}，"
                    f"水位线 id={watermark}）：不写库、不删缓冲"
                )
                return
            saved = 0
            # 昵称 → QQ 映射（本次缓冲里的真实发言者）：LLM 只回昵称，这里把昵称认回 QQ，
            # 作为画像的分组锚点，免得对方一改昵称就另开一行档案。
            nick2qq: dict[str, str] = {}
            for r in msgs:
                n = (r["user_name"] or "").strip()
                u = str(r["user_id"] or "").strip()
                if n and u and u != "system":
                    nick2qq[n] = u
            # 人物画像
            for p in data.get("profiles") or []:
                uname = str(p.get("user", "")).strip() or "未知"
                attrs = p.get("attrs") or {}
                if not attrs:
                    continue
                qq = _resolve_qq(uname, nick2qq) if self._identity_mode() == "qq" else ""
                # ★ 2026-09-30：改成**按人合并写入**。原来是无脑 INSERT 一行新快照 →
                #   同一个人的画像越堆越多（私聊实测 31 行 / 7,527 字，每轮注入的
                #   「提问者本人档案」里 3 行有 2/3 是重复内容）。合并后同一人只留一行。
                self._merge_profile(self_id, group_id, uname, attrs, qq=qq)
                saved += 1
            # 要点记忆
            # 要点记忆：user_name 用中性标签（原来写的是 bot 自己的名字，
            # 注入时每条都显示「(10-11 绫地宁宁)」，模型容易以为是它自己说过的话）
            point_label = "对话记录" if str(group_id).startswith("priv_") else "群聊记录"
            for m in data.get("memories") or []:
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                self._add_long(
                    self_id, group_id, "system", point_label, content[:200],
                    m.get("keywords") or [], "要点提取",
                )
                saved += 1
            # 清空缓冲（只删这一批，整理期间新到的消息留着下轮）
            self._clear_short(self_id, group_id, watermark)
            self._organize_backoff.pop(job_key, None)
            self._log(f"[{self._log_tag(self_id)}] 群 {group_id} 整理完成，新增 {saved} 条长期记忆")
        except Exception as e:
            self._organize_backoff[job_key] = time.time()
            self._log(f"[{self._log_tag(self_id)}] AI 整理失败: {e}")
        finally:
            self._organizing.discard(job_key)
            # 只释放自己的占位（别把接管后的新实例的占位删掉）
            if self._claim_is_mine(claims, job_key, token):
                claims.pop(job_key, None)

    # ---------------- bot 名字（按 bot 自动识别，代码里不写死角色名） ----------------
    def _plugin_display_name(self) -> str:
        """插件展示名（最后的兜底标签）。"""
        try:
            meta = self.context.get_registered_star("astrbot_plugin_stardust_diary")
            name = getattr(meta, "display_name", None) or getattr(meta, "name", None)
            if name:
                return str(name)
        except Exception:
            pass
        return "astrbot_plugin_stardust_diary"

    def _platform_instance_names(self) -> list[str]:
        """AstrBot 里配置的平台实例名。

        用户给每个 bot 起的实例名，也正是 unified_msg_origin 的前缀
        （`平台实例名:消息类型:会话号`），多 bot 时天然区分（甲 / 乙）。
        """
        try:
            insts = list(self.context.platform_manager.platform_insts or [])
        except Exception:
            insts = []
        out: list[str] = []
        for inst in insts:
            try:
                n = str(getattr(inst.meta(), "id", "") or "").strip()
            except Exception:
                continue
            if n and n not in out:
                out.append(n)
        return out

    def _configured_platform_names(self) -> list[str]:
        """AstrBot 全局配置里的所有平台实例名（**含未启用的**）。

        另一个 bot 平时可能是关着的（比如暂时停用的那个），但它的记忆库还在，
        先把名字认下来，画像键归一 / 启动日志才不会漏。
        """
        try:
            plats = self.context.get_config().get("platform", []) or []
        except Exception:
            return []
        out: list[str] = []
        for p in plats:
            if isinstance(p, dict):
                n = str(p.get("id", "") or "").strip()
            else:
                n = str(getattr(p, "id", "") or "").strip()
            if n and n not in out:
                out.append(n)
        return out

    def _cfg_bot_names(self) -> dict[str, str]:
        """配置项 bot_names：`10001=甲, 10002=乙`（也支持列表）。"""
        raw = self.config.get("bot_names", "") or ""
        out: dict[str, str] = {}
        for item in _split_names(raw):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                out[k] = v
        return out

    def _known_bot_names(self) -> list[str]:
        """目前认得的所有 bot 名字（配置 + 平台实例 + 从消息学到）。"""
        out: list[str] = []
        for n in (
            list(self._cfg_bot_names().values())
            + self._platform_instance_names()
            + self._configured_platform_names()
            + list(self._bot_names.values())
        ):
            n = str(n or "").strip()
            if n and n not in out:
                out.append(n)
        return out

    def _bot_name(self, self_id=None) -> str:
        """**这个 bot 自己**的名字。

        顺序：`bot_names` 配置指定 > 从消息事件学到的平台实例名 >
        全局只有一个平台实例时就用它 > 插件展示名。
        """
        sid = str(self_id or "").strip()
        if sid:
            n = self._cfg_bot_names().get(sid) or self._bot_names.get(sid)
            if n:
                return str(n)
        names = self._platform_instance_names()
        if len(names) == 1:
            return names[0]
        return self._plugin_display_name()

    def _refresh_profile_names(self) -> None:
        """把自动识别到的 bot 名字并进「画像别名名字」（手填的 profile_names 不动）。"""
        add_profile_names(
            list(self._cfg_bot_names().values())
            + self._platform_instance_names()
            + self._configured_platform_names()
        )

    def _learn_bot_name(self, self_id: str, event) -> None:
        """从消息事件学「这个 QQ 的 bot 叫什么」——UMO 前缀就是平台实例名。"""
        try:
            sid = str(self_id or "").strip()
            umo = str(getattr(event, "unified_msg_origin", "") or "")
            name = umo.split(":", 1)[0].strip()
            if not sid or not name or name == sid:
                return
            if self._bot_names.get(sid) != name:
                self._bot_names[sid] = name
                add_profile_names([name])
                self._log(f"[{self._log_tag(sid)}] 识别到 bot 名字: {name}（self_id={sid}）")
        except Exception as e:
            logger.debug(f"[SmartMemory] 识别 bot 名字失败（忽略）: {e}")

    def _log_tag(self, self_id=None) -> str:
        """日志/落库标签：这个 bot 自己的名字，取不到才退回插件展示名。"""
        return self._bot_name(self_id)

    LOG_MAX_BYTES = 1_000_000   # plugin.log 超过这么大就轮转一份（只留一代）

    def _log(self, msg: str):
        """写插件自己的日志文件（astrbot.log 会被 group_log_archive 定期清空）。

        超过 LOG_MAX_BYTES 就轮转成 plugin.log.1，避免无限增长。
        """
        path = os.path.join(self.db_dir, "plugin.log")
        try:
            if os.path.exists(path) and os.path.getsize(path) > self.LOG_MAX_BYTES:
                os.replace(path, path + ".1")
        except Exception:
            pass
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    def _get_scope(self, event: AstrMessageEvent) -> str:
        """作用域：群聊用群号，私聊用 priv_<对方QQ>，互不串扰。"""
        mtype = event.get_message_type()
        if mtype == MessageType.GROUP_MESSAGE:
            return str(event.get_group_id())
        if mtype == MessageType.FRIEND_MESSAGE:
            return "priv_" + str(event.get_sender_id())
        return "priv_" + str(event.get_sender_id())

    @staticmethod
    def _parse_json(s: str):
        """从 LLM 输出中提取 JSON 对象（容忍 markdown 包裹或前后废话）。"""
        s = (s or "").strip()
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            s = m.group(0)
        try:
            return json.loads(s)
        except Exception:
            return None

    def _pick_provider(self):
        """按配置选择 AI 整理用的供应商；未配置则用当前对话模型。"""
        try:
            pid = str(self.config.get("organize_provider", "") or "")
            if pid:
                p = self.context.get_provider_by_id(pid)
                if p is not None:
                    return p
                logger.warning(f"[{self._log_tag()}] 供应商 {pid} 不存在，改用当前模型")
            return self.context.get_using_provider()
        except Exception as e:
            logger.warning(f"[{self._log_tag()}] 选择供应商失败: {e}")
            return None

    def _save_plugin_config(self) -> None:
        """把当前 self.config 写回插件配置文件（指令修改配置时用）。

        优先走 `AstrBotConfig.save_config()`：它带写入版本号 + 临时文件原子替换，
        能和 WebUI 的保存串行化，避免"后写的静默覆盖先写的"。失败才退回手写。
        """
        save = getattr(self.config, "save_config", None)
        if callable(save):
            try:
                save()
                return
            except Exception as e:
                logger.warning(f"[{self._log_tag()}] save_config() 失败，退回手写: {e}")
        try:
            cfg_path = os.path.join(
                get_astrbot_data_path(), "config",
                "astrbot_plugin_stardust_diary_config.json",
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[{self._log_tag()}] 保存插件配置失败: {e}")

    def _count_short(self, self_id: str, group_id: str) -> int:
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM short_term WHERE group_id = ? AND expire_at > ?",
                    (group_id, time.time()),
                ).fetchone()[0]
        except Exception:
            return 0

    def _all_short(self, self_id: str, group_id: str, limit: int = 500):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND expire_at > ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (group_id, time.time(), limit),
                ).fetchall()
        except Exception:
            return []

    def _short_watermark(self, msgs) -> int:
        """这批消息里最大的 id；清缓冲时只删到它为止。"""
        try:
            return max(int(r["id"]) for r in msgs)
        except Exception:
            return 0

    def _clear_short(self, self_id: str, group_id: str, upto_id: int | None = None):
        """清缓冲。

        `upto_id` 非空时**只删到那条为止**：整理要跑十几秒，这期间新消息持续
        进缓冲，一刀切会把它们和已处理的一起删掉（静默丢记忆）。
        """
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                if upto_id is None:
                    # 显式要求"清空"（旧行为）；正常整理路径一律走下面的水位线分支
                    conn.execute(
                        "DELETE FROM short_term WHERE group_id = ?", (group_id,)
                    )
                else:
                    # upto_id 取不到（=0）时这条删不到任何行 —— 宁可不删，也不能误删新消息
                    conn.execute(
                        "DELETE FROM short_term WHERE group_id = ? AND id <= ?",
                        (group_id, int(upto_id)),
                    )
        except Exception as e:
            logger.warning(f"[{self._log_tag(self_id)}] 清空缓冲失败: {e}")

    # ---------------- 存储 ----------------
    def _add_short(self, self_id, group_id, user_id, user_name, text):
        now = time.time()
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                conn.execute(
                    "INSERT INTO short_term (group_id, user_id, user_name, content, created_at, expire_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (group_id, user_id, user_name, text[:500], now,
                     now + max(1, int(self.config.get("short_retain_days", 3))) * 86400),
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 短期记忆写入失败: {e}")
        # 过期行原本只在插件启动时清一次，长跑时短期表会一直涨、过期消息也还在被读。
        # 这里顺手每小时物理清一次（按 bot 库）。
        if now - self._last_purge > 3600:
            self._last_purge = now
            self._cleanup_expired(self._db_path(self_id))

    # 画像按人合并：同一份资料只留一行（新旧值同 key 时新值覆盖旧值），并限制长度。
    _PROFILE_SUFFIX_RE = re.compile(r"[（(][^（()）]*[)）]\s*$")
    PROFILE_MAX_KEYS = 24
    PROFILE_MAX_CHARS = 900
    ORGANIZE_FAIL_BACKOFF = 120   # 整理失败后，本群多少秒内不再重试（防按条烧 token）
    ORGANIZE_CLAIM_TTL = 300      # 跨实例整理占位的有效期；超时视为上个实例已崩，回收再占

    @classmethod
    def _profile_key(cls, uname: str) -> str:
        """'昵称(1234567890)' → '昵称'（去掉尾部括号备注，避免同一人开两份档案）。"""
        # 顺手把空白（含模型偶尔写进来的换行）压成单个空格：
        # 实测有人的画像名里带了换行，注入时那条画像会被劈成两半
        n = re.sub(r"\s+", " ", (uname or "")).strip()
        return (cls._PROFILE_SUFFIX_RE.sub("", n).strip() or n or "未知")

    @staticmethod
    def _rows_to_merged(rows) -> dict:
        """把若干画像行按键值合并（按传入顺序，靠后的值胜出）。同义键归一化。"""
        merged: dict[str, str] = {}
        for r in rows:
            body = r["content"] or ""
            body = body.split("：", 1)[1] if "：" in body else body
            for part in body.split("；"):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k, v = _canon_key(k), v.strip()
                if k and v:
                    merged[k] = v
        return merged

    def _merge_identity_rows(
        self, conn, identity: str, rows, merged: dict | None = None,
        kv_ts_new: dict | None = None,
    ) -> int:
        """把同一身份的画像行合并成一行（调用方负责事务）。返回删除的行数。

        `merged` 是调用方已经算好的键值（行里的旧值 + 本次新提炼的），它**必须**
        参与最终写入——早期版本这里只从 `rows` 重算，导致 `_merge_profile` 里辛苦
        合进去的新键值被整个丢掉（新建的人永远是一行空壳、老人的画像永不更新）。

        `kv_ts_new` 是本次新提炼的键 → 时间戳；旧的从各行的 kv_ts 列继承
        （行按 created_at 升序传入，后者覆盖前者）。只有出现在本批里的键才刷新时间，
        其余键的"最后一次被更新"保持原样——这正是动态键 TTL 能真正生效的前提。
        """
        work = self._rows_to_merged(rows)
        if merged:
            work.update(merged)
        if not work:
            return 0
        kv_ts = {}
        for r in rows:
            try:
                raw = r["kv_ts"]
            except (IndexError, KeyError):
                raw = ""
            for k, v in _kv_ts_load(raw).items():
                kv_ts[k] = v
        for k, v in (kv_ts_new or {}).items():
            kv_ts[k] = v
        kv_ts = {k: v for k, v in kv_ts.items() if k in work}   # 顺手清掉已不在的键
        content = _pack_profile(
            identity, work, self.PROFILE_MAX_KEYS, self.PROFILE_MAX_CHARS,
            created_at=rows[-1]["created_at"] if "created_at" in rows[-1].keys() else None,
            ttl_days=0,  # 合并的落盘长度不受 TTL 影响，TTL 只在注入前应用
        )
        kws = ",".join([identity] + [k for k, _ in _sorted_pairs(work)])[:200]
        latest = rows[-1]["id"]
        conn.execute(
            "UPDATE long_term SET user_name=?, content=?, keywords=?, created_at=?, kv_ts=? "
            "WHERE id=?",
            (identity, content, kws, time.time(), _kv_ts_dump(kv_ts), latest),
        )
        ids = [r["id"] for r in rows if r["id"] != latest]
        if ids:
            conn.executemany("DELETE FROM long_term WHERE id=?", [(i,) for i in ids])
        return len(ids)

    def _run_canonical_pass(self, conn, group_id: str) -> int:
        """把「同义键各占一份」的画像行重写成归一化键。

        幂等：重写后的内容再次归一化结果不变，所以不需要任何标记位——
        跑一次发现没有差异就是空操作（成本只有一次 SELECT）。

        Returns:
            被重写的行数。
        """
        changed = 0
        rows = conn.execute(
            "SELECT id, user_name, content, created_at, kv_ts FROM long_term "
            "WHERE group_id=? AND content LIKE '【画像】%'",
            (group_id,),
        ).fetchall()
        for r in rows:
            ident = _profile_identity(r["content"] or "", r["user_name"] or "")
            merged = self._rows_to_merged([r])
            if not merged:
                continue
            content = _pack_profile(
                ident, merged, self.PROFILE_MAX_KEYS, self.PROFILE_MAX_CHARS,
                created_at=r["created_at"],
                ttl_days=0,  # 归一化落盘不删动态键（保留数据），只在注入前按 TTL 过滤
            )
            # kv_ts 的键名跟着归一化后的键名走（`昵称`→`称呼` 这类改名不能丢时间戳）
            kv_ts = {}
            for k, v in _kv_ts_load(r["kv_ts"] if "kv_ts" in r.keys() else "").items():
                kv_ts[_canon_key(k)] = v
            kv_ts = {k: v for k, v in kv_ts.items() if k in merged}
            kv_raw = _kv_ts_dump(kv_ts)
            if content == (r["content"] or "") and kv_raw == (r["kv_ts"] or ""):
                continue
            kws = ",".join([ident] + [k for k, _ in _sorted_pairs(merged)])[:200]
            conn.execute(
                "UPDATE long_term SET user_name=?, content=?, keywords=?, kv_ts=? WHERE id=?",
                (ident, content, kws, kv_raw, r["id"]),
            )
            changed += 1
        if changed:
            logger.info("[SmartMemory] 画像同义键归一 %s：重写 %d 行", group_id, changed)
            self._log(f"[画像归一] {group_id}: 同义键归一 {changed} 行")
        return changed

    def _consolidate_profiles(self, db_path: str, group_id: str) -> int:
        """把一个作用域里同一人的多行画像合并成一行（幂等，自动执行）。

        典型场景：历史遗留的「昵称（旧档）」快照与后来正名的「昵称」各占一行，
        注入时一次带上好几段重复档案，把当轮对话语境挤掉。这里按键值合并：
        最新的行胜出；旧行独有的键继续保留；结果统一成一行并刷新 created_at。

        Returns:
            被删除的冗余行数（0 表示无需合并/已处理过）。
        """
        scoped_key = (db_path, group_id)
        if scoped_key in self._consolidated:
            return 0
        self._consolidated.add(scoped_key)
        if not bool(self.config.get("auto_merge_profiles", True)):
            return 0
        try:
            with closing(self._connect(db_path)) as conn, conn:
                self._run_canonical_pass(conn, group_id)
                rows = conn.execute(
                    "SELECT id, user_name, content, created_at, kv_ts FROM long_term "
                    "WHERE group_id=? AND content LIKE '【画像】%' "
                    "ORDER BY created_at ASC, id ASC",
                    (group_id,),
                ).fetchall()
                # ★ 分组按 QQ（`_profile_group_key`），不按昵称：昵称改了就换一行档案的坑
                groups: dict[str, list] = {}
                for r in rows:
                    gk = self._profile_group_key(r["content"] or "", r["user_name"] or "")
                    groups.setdefault(gk, []).append(r)
                removed = 0
                for _gk, items in groups.items():
                    if len(items) <= 1:
                        continue
                    # 显示名取最新那行的昵称（分组键可能是 "qq:123" 这种，不能当名字）
                    newest = items[-1]
                    ident = _profile_identity(newest["content"] or "", newest["user_name"] or "")
                    removed += self._merge_identity_rows(conn, ident, items)
                if removed:
                    logger.info(
                        "[SmartMemory] 画像自动合并 %s: %d 行 → %d 人（删除冗余 %d 行）",
                        group_id, len(rows), len(groups), removed,
                    )
                    self._log(
                        f"[画像合并] {group_id}: {len(rows)} 行 → {len(groups)} 人，删除 {removed} 行"
                    )
                return removed
        except Exception as e:
            logger.warning(f"[SmartMemory] 画像合并失败 {group_id}: {e}")
            return 0

    def _identity_mode(self) -> str:
        """画像认人方式：`qq`（默认，按 QQ）或 `name`（退回只按昵称，1.6.2 的旧行为）。"""
        mode = str(self.config.get("profile_identity_mode", "qq") or "qq").strip().lower()
        return mode if mode in ("qq", "name") else "qq"

    def _profile_group_key(self, content: str, user_name: str) -> str:
        """画像分组键（受 `profile_identity_mode` 控制）。"""
        if self._identity_mode() == "name":
            return "name:" + self._profile_key(user_name)
        return _profile_group_key(content, user_name)

    def _merge_profile(self, self_id, group_id, user_name, attrs, qq: str = ""):
        """把这次提炼的 attrs 合并进该人已有的画像行（没有就新建）。只留一行。

        - 同人识别**优先用 QQ**（`qq` 参数，或正文里已有的 `QQ=`）；没有 QQ 时才退回
          「归一化昵称」（去掉尾部括号备注）。只按昵称认人的话，对方一改昵称就会被
          当成新的人、另开一行档案（线上实测有人裂成 7 行）；
        - 双方 QQ 都有且不同 → 明确是别人，即使昵称撞了也不合并；
        - 写完顺手把该身份的历史重复行一并合并（不再依赖启动时的一次性扫描）；
        - 键按"新值覆盖旧值"，键数/总字数超上限时丢最早的键（那些事实通常也在要点记忆里）。
        """
        key = self._profile_key(user_name)
        qq = str(qq or "").strip()
        if self._identity_mode() == "name":
            qq = ""            # 配置成按昵称认人时，不写 QQ 锚点、也不按 QQ 分组
        attrs = dict(attrs or {})
        if qq:
            # 把 QQ 写进正文：它是之后分组的锚点（也是 90 档高优先键，不会被裁掉）
            attrs["QQ"] = qq
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                cands = conn.execute(
                    "SELECT id, user_name, content, created_at, kv_ts FROM long_term "
                    "WHERE group_id=? AND content LIKE '【画像】%' "
                    "ORDER BY created_at ASC, id ASC LIMIT 500",
                    (group_id,),
                ).fetchall()
                same = []
                for c in cands:
                    row_qq = _profile_qq(c["content"] or "")
                    row_name = _profile_identity(c["content"] or "", c["user_name"] or "")
                    if qq and row_qq:
                        if row_qq == qq:
                            same.append(c)      # QQ 相同 → 同一人（昵称改了也认）
                        # QQ 不同 → 是别人，昵称撞了也不并
                    elif row_name == key:
                        same.append(c)          # 至少一边没有 QQ，只能按昵称认
                merged = self._rows_to_merged(same)
                kv_ts_new = {}
                now_ts = time.time()
                for k, v in attrs.items():
                    k, v = _canon_key(str(k)), str(v).strip()
                    if k and v:
                        merged[k] = v
                        # ★ 只给本批出现的键打时间戳：动态键 TTL 靠它判断"这条还新不新"
                        kv_ts_new[k] = now_ts
                if not merged:
                    return False
                rows = list(same)
                if not rows:
                    # 该身份第一次建档：先插入一行占位，再走统一合并写回
                    conn.execute(
                        "INSERT INTO long_term (group_id, user_id, user_name, content, keywords, raw, created_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (group_id, "system", key, f"【画像】{key}：", "", "人物画像", time.time()),
                    )
                    placeholder = conn.execute(
                        "SELECT id, user_name, content, created_at, kv_ts FROM long_term "
                        "WHERE group_id=? AND content=? ORDER BY id DESC LIMIT 1",
                        (group_id, f"【画像】{key}："),
                    ).fetchone()
                    rows = [placeholder] if placeholder else []
                if not rows:
                    return False
                # ★ merged 必须传进去：只传 rows 的话，本次新提炼的键值会被
                #   _merge_identity_rows 内部的"从 rows 重算"覆盖掉（老 bug，
                #   表现为新人画像永远是空壳、老人画像永不更新）
                self._merge_identity_rows(conn, key, rows, merged, kv_ts_new)
                return True
        except Exception as e:
            logger.warning(f"[SmartMemory] 画像合并写入失败，回退为新增: {e}")
            try:
                kv = "；".join(f"{k}={v}" for k, v in (attrs or {}).items())
                self._add_long(self_id, group_id, "system", key, f"【画像】{key}：{kv}",
                               [key] + list((attrs or {}).keys()), "人物画像")
                return True
            except Exception:
                return False

    def _add_long(self, self_id, group_id, user_id, user_name, summary, keywords, raw):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                conn.execute(
                    "INSERT INTO long_term (group_id, user_id, user_name, content, keywords, raw, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        group_id,
                        user_id,
                        user_name,
                        summary,
                        ",".join(keywords[:8]),
                        raw[:500],
                        time.time(),
                    ),
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 长期记忆写入失败: {e}")

    # ---------------- 检索 ----------------
    # 检索扫描窗口（按 created_at 倒序取这么多行再打分）。
    # 原为 600：实测某群长期记忆已 3647 行，等于**最近 600 条以外的老记忆永远检索不到**。
    # 实测成本：扫 600 行 ≈1.9ms / 3647 行 ≈15ms（单次查询，本机），所以放宽到 5000 几乎无感。
    SEARCH_SCAN_LIMIT = 5000

    def _search_long(self, self_id: str, group_id: str, query: str, top_k: int = 5):
        query = (query or "").strip()
        q_norm = _norm(query)
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                rows = conn.execute(
                    "SELECT * FROM long_term WHERE group_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (group_id, self.SEARCH_SCAN_LIMIT),
                ).fetchall()
        except Exception as e:
            logger.warning(f"[SmartMemory] 检索失败: {e}")
            return []
        scored = []
        time_level = self._detect_time_intent(query)
        time_since = self._since_ts(time_level) if time_level else 0.0
        # ★ 纯语气/寒暄的短消息不做长期记忆召回。
        #   实测：「好~」会靠"关键词命中"（kw in query）捞出 5 条/380 字与当下无关的旧记忆
        #   （因为很多记忆的关键词就是「好~」「嘿嘿~」这类口头禅），而正文 gram 命中为 0。
        #   对方口头禅频出的场景下这个坑很深，所以整条召回直接跳过。
        weak_query = _is_weak_query(query) and group_id.startswith("priv_")
        if weak_query:
            logger.debug("[SmartMemory] 纯语气/寒暄消息，跳过长期记忆召回：%r", query)
            return []
        for r in rows:
            content = r["content"] or ""
            uname = r["user_name"] or ""
            # ★ 画像行 / 带保质期的「旧档」内容不参与普通关键词召回：
            #   画像里常写着一堆口头禅，短句「唔....」会被 4-gram 打高分捞出来，
            #   于是每轮都注入好几段过时档案，把当轮语境挤掉。
            if _is_profile(content, uname) or _is_temp_recall(content, uname):
                continue
            score = 0
            text_hit = False
            c_norm = _norm(content)
            # 关键词：原文与简体化后各命中一次（原/简混合也能对上）
            for kw in (r["keywords"] or "").split(","):
                if kw and (kw in query or _norm(kw) in q_norm):
                    score += 2
                    text_hit = True
            # 正文包含提问原文 / 提问的简体形式
            if query and (query in content or q_norm in c_norm):
                score += 3
                text_hit = True
            # 4-gram 重叠（原文 + 简体化各算一遍），至少两个 gram 才算数，避免语气词擦边
            grams = self._grams(query) | self._grams(q_norm)
            gram_hits = sum(1 for gram in grams if gram and (gram in content or gram in c_norm))
            if gram_hits:
                score += min(gram_hits, 3)
                if gram_hits >= 2:
                    text_hit = True
            # 时间意图命中：只给「本来就有文本相关性」的记忆加权，
            # 绝不把毫不相关的行凭空塞进榜（原来 score==0 也 +2）
            if time_since and text_hit and (r["created_at"] or 0) >= time_since:
                score += 2
            min_score = max(1, int(self.config.get("min_score", 2) or 2))
            if score >= min_score:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        # 画像类记忆最多占少量名额，别把普通要点挤光
        max_profiles = max(0, min(4, int(self.config.get("max_profile_memories", 1) or 0)))
        picked, used_profiles = [], 0
        for _s, r in scored:
            if len(picked) >= top_k:
                break
            if _is_profile(r["content"] or "", r["user_name"] or ""):
                if used_profiles >= max_profiles:
                    continue
                used_profiles += 1
            picked.append(r)
        return picked

    @staticmethod
    def _grams(s: str, n: int = 4):
        s = re.sub(r"\s+", "", s or "")
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

    def _member_rows(self, self_id: str, group_id: str, user_id, user_name, limit: int = 4):
        """按 QQ号/昵称 检索某成员本人的画像与长期记忆（兜底注入用）。

        画像行 user_id 通常是 system，真实身份写在 content（QQ号=xxx）里，
        因此优先用 content 内的 QQ 号匹配，其次用昵称。
        """
        uid = str(user_id or "")
        name = (user_name or "").strip()
        conds, args = [], [group_id]
        if uid:
            # ★ QQ 在正文里有好几种写法：1.6.3 起统一写 `QQ=<uid>`，老数据里可能是
            #   `QQ号=` / 全角冒号 / `昵称(uid)`。以前只匹配后两种，于是**换了昵称的人
            #   明明有画像也找不到**（实测：某成员画像里写着 QQ=123456789，却一条都取不出来）。
            u = _like_escape(uid)
            for pat in (f"%QQ={u}%", f"%QQ号={u}%", f"%QQ：{u}%",
                        f"%QQ:{u}%", f"%({u})%"):
                conds.append("content LIKE ? ESCAPE '\\'")
                args.append(pat)
            conds.append("user_id = ?")
            args.append(uid)
        if name:
            # ★ 昵称必须转义：不转义时昵称里的 % / _ 会当通配符，
            #   把别人的画像当成"本人"返回（实测传 '%' 会命中整个库的画像行）
            conds.append("user_name LIKE ? ESCAPE '\\'")
            args.append(f"%{_like_escape(name)}%")
        if not conds:
            return []
        args.append(limit)
        sql = (
            "SELECT * FROM long_term WHERE group_id = ? AND ("
            + " OR ".join(conds)
            # ★ 2026-09-30：降级的历史快照（user_name 带「旧档」）不再占「本人档案」这一栏，
            #   否则同一份资料会以 3~4 行的形式挤进每轮提示词。它们仍在库里、检索照样能捞。
            + ") AND (user_name IS NULL OR user_name NOT LIKE '%旧档%')"
            " ORDER BY (content LIKE '【画像】%') DESC, created_at DESC LIMIT ?"
        )
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(sql, tuple(args)).fetchall()
        except Exception as e:
            logger.warning(f"[SmartMemory] 成员档案检索失败: {e}")
            return []

    def _since_ts(self, level: str) -> float:
        """分层时间起点：L1长期全部 / L2今天 / L3最近三天 / L4本周一。"""
        now = datetime.now()
        if level == "L2":
            return datetime(now.year, now.month, now.day).timestamp()
        if level == "L3":
            return (now - timedelta(days=3)).timestamp()
        if level == "L4":
            week_start = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return week_start.timestamp()
        return 0.0  # L1：全部长期记忆

    def _detect_time_intent(self, query: str) -> str | None:
        """根据提问内容判断时间层级意图。"""
        if any(k in query for k in ("今天", "刚刚", "刚才", "此刻", "现在", "今早", "今晨",
                                    "今晚", "今夜", "今儿", "早上", "上午", "下午", "中午", "晚上")):
            return "L2"
        if any(k in query for k in ("最近", "这两天", "这几天", "前两天", "前天", "大前天",
                                    "昨天", "昨晚", "昨日", "昨夜")):
            return "L3"
        if any(k in query for k in ("本周", "这周", "这星期", "这个星期", "星期", "一周", "上周")):
            return "L4"
        return None

    @staticmethod
    def _now_str() -> str:
        """当前时间锚点，供模型换算“今天/昨天/最近”。"""
        now = datetime.now()
        week = "一二三四五六日"[now.weekday()]
        return now.strftime("%Y-%m-%d %H:%M") + f" 周{week}"

    def _cfg_int(self, key: str, default: int, lo: int, hi: int) -> int:
        """读一个整数配置并夹到 [lo, hi]，坏值回退默认。"""
        try:
            val = int(self.config.get(key, default))
        except (TypeError, ValueError):
            val = default
        return max(lo, min(hi, val))

    def _profile_for_inject(self, row, cap: int) -> str:
        """把一条画像行整理成注入用的短文本。

        与 store 的区别（关键）：这里按**优先级**排序并按 cap 丢最低优先级的键，
        而不是像 _clip 那样直接砍尾巴——否则字母序靠前的高价值键会被保住，
        「身份/关系/擅长」这类反而被截掉。动态键按 TTL 过滤。
        """
        raw = row["content"] or ""
        ident = _profile_identity(raw, row["user_name"] or "")
        merged = self._rows_to_merged([row])
        if not merged:
            return ""          # 空壳画像（没有任何键值）没有注入价值
        ttl_days = self._cfg_int("profile_dynamic_ttl_days", 3, 0, 365)
        try:
            created_at = row["created_at"]
        except Exception:
            created_at = None
        try:
            kv_ts = _kv_ts_load(row["kv_ts"])
        except Exception:
            kv_ts = {}
        body = _pack_profile(
            ident, merged, self.PROFILE_MAX_KEYS, cap,
            created_at=created_at, ttl_days=ttl_days, kv_ts=kv_ts,
        )
        # 动态键被 TTL 全丢光时 _pack_profile 会剩一个「【画像】甲：」的空壳，直接当没有
        return body if "=" in body else ""

    def _recent_short_since(self, self_id: str, group_id: str, since_ts: float, n: int = 10):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND created_at >= ? "
                    "AND expire_at > ? ORDER BY created_at DESC LIMIT ?",
                    (group_id, since_ts, time.time(), n),
                ).fetchall()
        except Exception:
            return []

    def _recent_short(self, self_id: str, group_id: str, n: int = 10):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND expire_at > ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (group_id, time.time(), n),
                ).fetchall()
        except Exception:
            return []

    # ---------------- 注入上下文 ----------------
    @filter.on_llm_request()
    async def on_llm_req(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            if event.get_message_type() not in (MessageType.GROUP_MESSAGE, MessageType.FRIEND_MESSAGE):
                return
            group_id = self._get_scope(event)
            self_id = str(event.get_self_id())
            query = event.get_message_str().strip()
            if not query:
                return
            self._ensure_db(self_id)
            parts = []
            # 相关长期记忆
            top_k = int(self.config.get("top_k", 5))
            mems = self._search_long(self_id, group_id, query, top_k)
            if mems:
                lines = []
                for r in mems:
                    when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                    lines.append(f"- ({when} {r['user_name']}) {r['content']}")
                scope_label = "本会话" if str(group_id).startswith("priv_") else "本群"
                parts.append(
                    f"【记忆片段，来自{scope_label}历史消息，可参考但勿编造】\n" + "\n".join(lines)
                )
            # 提问者本人档案/记忆：默认「被试及到自己 / 探询 / 没召回到东西」时带上本人画像。
            # ★ 以前只看「探询正则 或 没召回到东西」，于是**只要召回到记忆就永远不注入画像**
            #   （群里几乎每轮都能召回到东西 → 画像长期不出现，表现为"它好像不认识我"）。
            mode = str(self.config.get("profile_inject_mode", "self") or "self").strip().lower()
            if mode not in ("off", "fallback", "self", "always"):
                mode = "self"
            probe = bool(_PROBE_RE.search(query))
            self_ref = bool(_SELF_RE.search(query))
            if mode == "always":
                want_profile = True
            elif mode == "self":
                want_profile = probe or self_ref or not mems
            elif mode == "fallback":
                want_profile = probe or (not mems and bool(self.config.get("profile_fallback", True)))
            else:                       # off
                want_profile = False
            uid = str(event.get_sender_id() or "")
            uname = event.get_sender_name() or uid
            if uid and want_profile:
                try:
                    lim = self._cfg_int("profile_limit", 1, 1, 5)
                    cap = self._cfg_int("profile_max_chars", 80, 40, 600)
                    # ★ 问的是"他自己"时多给材料：实测「宁宁求锐评我」在 80 字上限下
                    #   只剩 4 条事实（身份/互动/口头禅…），模型只能泛泛地答
                    #   （回一句"话太少了"）。这类问题把上限放到 3 倍、并多取两行
                    #   （`_member_rows` 是"画像优先"，多取的行就是这个人自己的长期记忆）。
                    if probe or self_ref:
                        lim = min(5, lim + 2)
                        cap = min(400, cap * 3)
                    own = self._member_rows(self_id, group_id, uid, uname, lim)
                    if own:
                        dup = {m["id"] for m in (mems or [])}
                        own_lines = []
                        for r in own:
                            if r["id"] in dup:
                                continue
                            body = self._profile_for_inject(r, cap)
                            if not body:      # 空壳 / 动态键全过期 → 不带这条
                                continue
                            when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                            own_lines.append(f"- ({when} {r['user_name']}) {body}")
                        if own_lines:
                            parts.append("【提问者本人档案/记忆，涉及ta自己或你俩关系时以此为准】\n"
                                        + "\n".join(own_lines[:lim]))
                except Exception:
                    pass
            # 分层短期记忆：识别提问的时间意图（L2今天/L3最近几天/L4本周）
            # ★ 私聊不再注入「今天的消息记录」：私聊没被 @ 的门槛，bot 手里就是完整会话历史，
            #   这段记录是同一段对话的原样重复（实测「今天没有雨啦」被塞进 10 条/409 字，
            #   而它本来就在上下文里）。群聊仍然需要——群里 bot 只看得到被唤醒那几条。
            level = self._detect_time_intent(query)
            if level and group_id.startswith("priv_"):
                level = None
            if level:
                since = self._since_ts(level)
                n = int(self.config.get("recent_count", 5)) * 2
                rows = self._recent_short_since(self_id, group_id, since, n)
                label = {"L2": "今天", "L3": "最近三天", "L4": "本周"}.get(level, level)
                if rows:
                    lines = [
                        f"- {datetime.fromtimestamp(r['created_at']).strftime('%m-%d %H:%M')} "
                        f"{r['user_name']}: {r['content']}"
                        for r in reversed(rows)
                    ]
                    parts.append(
                        f"【{label}的消息记录，用户可能问的是这段时间的事】\n"
                        f"【当前时间】{self._now_str()}，请据此换算今天/昨天/最近\n"
                        + "\n".join(lines)
                    )
                # 关键词追问引导
                parts.append(
                    "【检索提示】若用户询问时间段内发生的事情但描述模糊，"
                    "请先引导用户说出更具体的关键词（人名/话题/事件），"
                    "再根据关键词检索上面的记忆和【记忆片段】后回答，不要凭空编造。"
                )
            elif (
                self.config.get("include_recent", True)
                and bool(self.config.get("inject_recent_list", False))
            ):
                n = int(self.config.get("recent_count", 5))
                recent = self._recent_short(self_id, group_id, n)
                if recent:
                    lines = [
                        f"- {r['user_name']}: {r['content']}" for r in reversed(recent)
                    ]
                    parts.append("【最近消息，可能与本条相关】\n" + "\n".join(lines))
            if parts:
                # 注入到用户消息之后而非 system_prompt，保持前缀稳定，缓存才能命中
                req.extra_user_content_parts.append(
                    TextPart(text="\n\n".join(parts))
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 注入记忆失败: {e}")

    # ---------------- 指令 ----------------
    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("mem", alias={"memory"})
    async def mem(self, event: AstrMessageEvent):
        # 全部子指令（list/recent/forget/models/model/stat/stat all/help）仅管理员可用
        if not event.is_admin():
            yield event.plain_result("这条指令只有管理员能用哦～")
            return
        args = (event.message_str or "").split()
        sub = args[1].strip().lower() if len(args) > 1 else "help"
        group_id = self._get_scope(event)
        self_id = str(event.get_self_id())
        user_id = str(event.get_sender_id())
        self._ensure_db(self_id)

        if sub == "list":
            n = int(args[2]) if len(args) > 2 and args[2].isdigit() else 10
            try:
                with closing(self._connect(self._db_path(self_id))) as conn:
                    rows = conn.execute(
                        "SELECT * FROM long_term WHERE group_id = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (group_id, n),
                    ).fetchall()
            except Exception as e:
                yield event.plain_result(f"查询失败: {e}")
                return
            if not rows:
                yield event.plain_result("本群还没有长期记忆哦～")
                return
            lines = [f"本群最近 {len(rows)} 条长期记忆："]
            for r in rows:
                when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d %H:%M")
                lines.append(f"[{r['id']}] ({when} {r['user_name']}) {r['content']}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "recent":
            n = int(args[2]) if len(args) > 2 and args[2].isdigit() else 10
            recent = self._recent_short(self_id, group_id, n)
            if not recent:
                yield event.plain_result("本群最近没有短期消息记录～")
                return
            lines = [f"本群最近 {len(recent)} 条消息（保留期内有效）："]
            for r in reversed(recent):
                when = datetime.fromtimestamp(r["created_at"]).strftime("%H:%M")
                lines.append(f"({when} {r['user_name']}) {r['content']}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "forget":
            if len(args) < 3 or not args[2].isdigit():
                yield event.plain_result("用法：/mem forget <id>")
                return
            mid = int(args[2])
            try:
                with closing(self._connect(self._db_path(self_id))) as conn, conn:
                    row = conn.execute(
                        "SELECT * FROM long_term WHERE id = ? AND group_id = ?",
                        (mid, group_id),
                    ).fetchone()
                    if not row:
                        yield event.plain_result("没有找到这条记忆（只能操作本群的）")
                        return
                    if str(row["user_id"]) != user_id and not event.is_admin():
                        yield event.plain_result("这不是你的记忆，只有管理员能删哦～")
                        return
                    conn.execute("DELETE FROM long_term WHERE id = ?", (mid,))
                yield event.plain_result(f"已删除记忆 [{mid}]：{row['content']}")
            except Exception as e:
                yield event.plain_result(f"删除失败: {e}")
            return

        if sub == "models":
            providers = self.context.get_all_providers()
            if not providers:
                yield event.plain_result("当前没有任何模型供应商哦～")
                return
            lines = ["可用模型供应商："]
            for p in providers:
                m = p.meta
                lines.append(f"- {m.id}（{m.type}）当前模型：{m.model}")
                try:
                    models = await p.get_models()
                    if models:
                        shown = "、".join(models[:10])
                        lines.append(f"  可选：{shown}")
                except Exception:
                    pass
            lines.append("用 /mem model <供应商id> 设置整理用提供商")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "model":
            if len(args) < 3:
                yield event.plain_result("用法：/mem model <供应商id>，先 /mem models 查看")
                return
            pid = args[2]
            self.config["organize_provider"] = pid
            self._save_plugin_config()
            yield event.plain_result(f"AI整理提供商已设为：{pid}")
            return

        if sub == "stat":
            try:
                # /mem stat all：多 bot 维度汇总每个库的长期/短期总数
                if len(args) > 2 and args[2].lower() in ("all", "bot", "bots"):
                    names = sorted(
                        n for n in os.listdir(self.db_dir)
                        if re.fullmatch(r"memory_[0-9A-Za-z_-]+\.db", n)
                    )
                    if not names:
                        yield event.plain_result("还没有任何 bot 的记忆库哦～")
                        return
                    lines = ["各 bot 记忆库统计："]
                    for n in names:
                        # 单库出错（半截文件 / 还没建表）不该让整条指令失败
                        try:
                            with closing(self._connect(os.path.join(self.db_dir, n))) as conn:
                                long_n = conn.execute(
                                    "SELECT COUNT(*) FROM long_term"
                                ).fetchone()[0]
                                short_n = conn.execute(
                                    "SELECT COUNT(*) FROM short_term WHERE expire_at > ?",
                                    (time.time(),),
                                ).fetchone()[0]
                            bot = n[len("memory_"):-len(".db")]
                            lines.append(f"bot {bot}: 长期 {long_n} 条，短期（保留期内）{short_n} 条")
                        except Exception as e:
                            lines.append(f"{n}: 读取失败（{e}）")
                    yield event.plain_result("\n".join(lines))
                    return
                with closing(self._connect(self._db_path(self_id))) as conn:
                    long_n = conn.execute(
                        "SELECT COUNT(*) FROM long_term WHERE group_id = ?", (group_id,)
                    ).fetchone()[0]
                    short_n = conn.execute(
                        "SELECT COUNT(*) FROM short_term WHERE group_id = ? AND expire_at > ?",
                        (group_id, time.time()),
                    ).fetchone()[0]
                yield event.plain_result(
                    f"本群记忆统计：长期 {long_n} 条，短期（保留期内）{short_n} 条"
                )
            except Exception as e:
                yield event.plain_result(f"统计失败: {e}")
            return

        yield event.plain_result(
            "Smart Memory 指令：\n"
            "/mem list [n] - 查看长期记忆\n"
            "/mem recent [n] - 查看最近消息\n"
            "/mem forget <id> - 删除记忆\n"
            "/mem stat - 统计\n"
            "/mem help - 帮助"
        )
