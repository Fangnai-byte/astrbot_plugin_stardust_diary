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
    """画像行归一化身份：优先取正文『【画像】X：』里的 X，否则退回 user_name。"""
    c = (content or "").lstrip()
    if c.startswith("【画像】"):
        rest = c[len("【画像】"):]
        for sep in ("：", ":"):
            if sep in rest:
                name = rest.split(sep, 1)[0].strip()
                if name:
                    return SmartMemory._profile_key(name)
    return SmartMemory._profile_key(user_name)


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
    # 90：核心身份与关系，永远优先
    90: ("身份", "身份补充", "关系", "称呼", "擅长", "语言", "身体", "昵称"),
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
    d: dict, created_at: float | None, ttl_days: int
) -> tuple[dict, list[str]]:
    """按 TTL 去掉过期的动态键。返回 (保留的字典, 被去掉的键名列表)。

    画像行的 created_at 只在内容被更新（合并/归一）时刷新，所以它是"这条画像
    最近一次被改"的可靠时间锚点；ttl_days <= 0 表示不做时效过滤。
    """
    if not ttl_days or ttl_days <= 0 or not created_at:
        return d, []
    age_days = (time.time() - float(created_at)) / 86400.0
    if age_days <= ttl_days:
        return d, []
    kept, dropped = {}, []
    for k, v in d.items():
        if k in PROFILE_DYNAMIC_KEYS:
            dropped.append(k)
        else:
            kept[k] = v
    return (kept or d), dropped


def _pack_profile(
    identity: str,
    d: dict,
    max_keys: int,
    max_chars: int,
    created_at: float | None = None,
    ttl_days: int = 0,
) -> str:
    """把键值字典打包成一行画像。

    超限时的取舍顺序（重要）：先按 TTL 去掉过期动态键 → 按优先级排序
    （同优先级按键名）→ 超字数时从**最低优先级**开始丢，而不是按字母序。
    """
    work = dict(d or {})
    if ttl_days:
        work, _dropped = _profile_drop_dynamic(work, created_at, ttl_days)
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
        self._organizing: set = set()    # 正在整理的群，避免并发重复触发
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
        self._init_all_db()              # 为已存在的各 bot 库建表并清理过期

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
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_long_group ON long_term(group_id)")
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
    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE
    )
    async def on_group_message(self, event: AstrMessageEvent):
        # 忽略 bot 自己的消息，避免自我循环
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        self_id = str(event.get_self_id())
        text = event.get_message_str().strip()
        if not text:
            return
        # 插件自身指令不参与记忆
        if text.startswith("/mem"):
            return
        group_id = self._get_scope(event)
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
        try:
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
            text = "\n".join(
                f"{r['user_name']}({r['user_id']}): {r['content']}" for r in msgs
            )[:12000]
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
            saved = 0
            # 人物画像
            for p in data.get("profiles") or []:
                uname = str(p.get("user", "")).strip() or "未知"
                attrs = p.get("attrs") or {}
                if not attrs:
                    continue
                # ★ 2026-09-30：改成**按人合并写入**。原来是无脑 INSERT 一行新快照 →
                #   同一个人的画像越堆越多（私聊实测 31 行 / 7,527 字，每轮注入的
                #   「提问者本人档案」里 3 行有 2/3 是重复内容）。合并后同一人只留一行。
                self._merge_profile(self_id, group_id, uname, attrs)
                saved += 1
            # 要点记忆
            for m in data.get("memories") or []:
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                self._add_long(
                    self_id, group_id, "system", self._log_tag(self_id), content[:200],
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

        另一个 bot 平时可能是关着的（如乙），但它的记忆库还在，
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

    @classmethod
    def _profile_key(cls, uname: str) -> str:
        """'昵称(1234567890)' → '昵称'（去掉尾部括号备注，避免同一人开两份档案）。"""
        n = (uname or "").strip()
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

    def _merge_identity_rows(self, conn, identity: str, rows, merged: dict | None = None) -> int:
        """把同一身份的画像行合并成一行（调用方负责事务）。返回删除的行数。

        `merged` 是调用方已经算好的键值（行里的旧值 + 本次新提炼的），它**必须**
        参与最终写入——早期版本这里只从 `rows` 重算，导致 `_merge_profile` 里辛苦
        合进去的新键值被整个丢掉（新建的人永远是一行空壳、老人的画像永不更新）。
        """
        work = self._rows_to_merged(rows)
        if merged:
            work.update(merged)
        if not work:
            return 0
        content = _pack_profile(
            identity, work, self.PROFILE_MAX_KEYS, self.PROFILE_MAX_CHARS,
            created_at=rows[-1]["created_at"] if "created_at" in rows[-1].keys() else None,
            ttl_days=0,  # 合并的落盘长度不受 TTL 影响，TTL 只在注入前应用
        )
        kws = ",".join([identity] + [k for k, _ in _sorted_pairs(work)])[:200]
        latest = rows[-1]["id"]
        conn.execute(
            "UPDATE long_term SET user_name=?, content=?, keywords=?, created_at=? "
            "WHERE id=?",
            (identity, content, kws, time.time(), latest),
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
            "SELECT id, user_name, content, created_at FROM long_term "
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
            if content == (r["content"] or ""):
                continue
            kws = ",".join([ident] + [k for k, _ in _sorted_pairs(merged)])[:200]
            conn.execute(
                "UPDATE long_term SET user_name=?, content=?, keywords=? WHERE id=?",
                (ident, content, kws, r["id"]),
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
                    "SELECT id, user_name, content, created_at FROM long_term "
                    "WHERE group_id=? AND content LIKE '【画像】%' "
                    "ORDER BY created_at ASC, id ASC",
                    (group_id,),
                ).fetchall()
                groups: dict[str, list] = {}
                for r in rows:
                    ident = _profile_identity(r["content"] or "", r["user_name"] or "")
                    groups.setdefault(ident, []).append(r)
                removed = 0
                for ident, items in groups.items():
                    if len(items) <= 1:
                        continue
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

    def _merge_profile(self, self_id, group_id, user_name, attrs):
        """把这次提炼的 attrs 合并进该人已有的画像行（没有就新建）。只留一行。

        - 同人识别用「归一化身份」（去掉尾部括号备注），所以 '昵称' 与
          '昵称(1234567890)'、'昵称（旧档）' 会归并到同一行；
        - 写完顺手把该身份的历史重复行一并合并（不再依赖启动时的一次性扫描）；
        - 键按"新值覆盖旧值"，键数/总字数超上限时丢最早的键（那些事实通常也在要点记忆里）。
        """
        key = self._profile_key(user_name)
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                cands = conn.execute(
                    "SELECT id, user_name, content, created_at FROM long_term "
                    "WHERE group_id=? AND content LIKE '【画像】%' "
                    "ORDER BY created_at ASC, id ASC LIMIT 500",
                    (group_id,),
                ).fetchall()
                same = [
                    c for c in cands
                    if _profile_identity(c["content"] or "", c["user_name"] or "") == key
                ]
                merged = self._rows_to_merged(same)
                for k, v in (attrs or {}).items():
                    k, v = _canon_key(str(k)), str(v).strip()
                    if k and v:
                        merged[k] = v
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
                        "SELECT id, user_name, content, created_at FROM long_term "
                        "WHERE group_id=? AND content=? ORDER BY id DESC LIMIT 1",
                        (group_id, f"【画像】{key}："),
                    ).fetchone()
                    rows = [placeholder] if placeholder else []
                if not rows:
                    return False
                # ★ merged 必须传进去：只传 rows 的话，本次新提炼的键值会被
                #   _merge_identity_rows 内部的"从 rows 重算"覆盖掉（老 bug，
                #   表现为新人画像永远是空壳、老人画像永不更新）
                self._merge_identity_rows(conn, key, rows, merged)
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
            conds.append("content LIKE ?")
            args.append(f"%QQ号={uid}%")
            conds.append("content LIKE ?")
            args.append(f"%({uid})%")
            conds.append("user_id = ?")
            args.append(uid)
        if name:
            conds.append("user_name LIKE ?")
            args.append(f"%{name}%")
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
            return _clip(raw, cap)
        ttl_days = self._cfg_int("profile_dynamic_ttl_days", 3, 0, 365)
        try:
            created_at = row["created_at"]
        except Exception:
            created_at = None
        return _pack_profile(
            ident, merged, self.PROFILE_MAX_KEYS, cap,
            created_at=created_at, ttl_days=ttl_days,
        )

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
                parts.append("【记忆片段，来自本群历史消息，可参考但勿编造】\n" + "\n".join(lines))
            # 提问者本人档案/记忆兜底：问“你记得我吗/我平时怎样”时本人画像优先
            uid = str(event.get_sender_id() or "")
            uname = event.get_sender_name() or uid
            if uid and (
                _PROBE_RE.search(query)
                or (not mems and bool(self.config.get("profile_fallback", True)))
            ):
                try:
                    lim = self._cfg_int("profile_limit", 1, 1, 5)
                    cap = self._cfg_int("profile_max_chars", 120, 40, 600)
                    own = self._member_rows(self_id, group_id, uid, uname, lim)
                    if own:
                        dup = {m["id"] for m in (mems or [])}
                        own_lines = []
                        for r in own:
                            if r["id"] in dup:
                                continue
                            when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                            body = self._profile_for_inject(r, cap)
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
