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


# 命中这些词的消息才值得交给 LLM 判断（节省 token）
class SmartMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_dir = get_astrbot_data_path()
        self.db_dir = os.path.join(data_dir, "smart_memory")
        os.makedirs(self.db_dir, exist_ok=True)
        self._legacy_db = os.path.join(self.db_dir, "memory.db")  # 旧版单库（升级后仅迁移一次）
        self._ensured: set = set()       # 已初始化（建表）的 bot 库
        self._organizing: set = set()    # 正在整理的群，避免并发重复触发
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

    @staticmethod
    def _schema(conn) -> None:
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
        """启动时为所有已存在的 memory_*.db 建表并清理过期。"""
        try:
            for name in os.listdir(self.db_dir):
                if re.fullmatch(r"memory_[0-9A-Za-z_-]+\.db", name):
                    p = os.path.join(self.db_dir, name)
                    try:
                        with closing(self._connect(p)) as conn, conn:
                            self._schema(conn)
                        self._cleanup_expired(p)
                        self._ensured.add(p)
                    except Exception as e:
                        logger.warning(f"[SmartMemory] 初始化库 {name} 失败: {e}")
        except Exception as e:
            logger.warning(f"[SmartMemory] 扫描数据库目录失败: {e}")

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
        self._add_short(self_id, group_id, user_id, user_name, text)

        # 2. 满阈值触发 AI 整理（异步，不阻塞回复）
        if self.config.get("organize_enabled", True):
            n = self._count_short(self_id, group_id)
            if n >= int(self.config.get("organize_threshold", 100)):
                asyncio.create_task(self._organize(self_id, group_id))

    # ---------------- 满阈值 AI 整理 ----------------
    async def _organize(self, self_id: str, group_id: str):
        """攒满阈值后：AI 提炼人物画像键值+要点记忆，存长期后清空缓冲。"""
        if group_id in self._organizing:
            self._log(f"[绫地宁宁] 群 {group_id} 正在整理中，跳过本次触发")
            return
        self._organizing.add(group_id)
        try:
            self._log(f"[绫地宁宁] _organize 被调用，群 {group_id}")
            msgs = self._all_short(self_id, group_id)
            self._log(f"[绫地宁宁] 群 {group_id} 缓冲 {len(msgs)} 条")
            if len(msgs) < int(self.config.get("organize_threshold", 100)):
                return
            text = "\n".join(
                f"{r['user_name']}({r['user_id']}): {r['content']}" for r in msgs
            )[:12000]
            provider = self._pick_provider()
            self._log(f"[绫地宁宁] provider: {provider.meta().id if provider else None}")
            if provider is None:
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
            self._log(f"[绫地宁宁] LLM 返回前100字: {out[:100]!r}")
            data = self._parse_json(out)
            self._log(f"[绫地宁宁] 解析结果: {bool(data)}")
            if not data:
                return
            saved = 0
            # 人物画像
            for p in data.get("profiles") or []:
                uname = str(p.get("user", "")).strip() or "未知"
                attrs = p.get("attrs") or {}
                if not attrs:
                    continue
                kv = "；".join(f"{k}={v}" for k, v in attrs.items())
                self._add_long(
                    self_id, group_id, "system", uname, f"【画像】{uname}：{kv}",
                    [uname] + list(attrs.keys()), "人物画像",
                )
                saved += 1
            # 要点记忆
            for m in data.get("memories") or []:
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                self._add_long(
                    self_id, group_id, "system", "绫地宁宁", content[:200],
                    m.get("keywords") or [], "要点提取",
                )
                saved += 1
            # 清空缓冲，重新计数
            self._clear_short(self_id, group_id)
            self._log(f"[绫地宁宁] 群 {group_id} 整理完成，新增 {saved} 条长期记忆")
        except Exception as e:
            self._log(f"[绫地宁宁] AI 整理失败: {e}")
        finally:
            self._organizing.discard(group_id)

    def _log(self, msg: str):
        """双写日志：astrbot 日志 + 独立文件（防 group_log_archive 清空源日志丢失）"""
        try:
            with open(os.path.join(self.db_dir, "plugin.log"), "a", encoding="utf-8") as f:
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
                logger.warning(f"[绫地宁宁] 供应商 {pid} 不存在，改用当前模型")
            return self.context.get_using_provider()
        except Exception as e:
            logger.warning(f"[绫地宁宁] 选择供应商失败: {e}")
            return None

    def _save_plugin_config(self) -> None:
        """把当前 self.config 写回插件配置文件（指令修改配置时用）"""
        try:
            cfg_path = os.path.join(
                get_astrbot_data_path(), "config",
                "astrbot_plugin_stardust_diary_config.json",
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[绫地宁宁] 保存插件配置失败: {e}")

    def _count_short(self, self_id: str, group_id: str) -> int:
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM short_term WHERE group_id = ?",
                    (group_id,),
                ).fetchone()[0]
        except Exception:
            return 0

    def _all_short(self, self_id: str, group_id: str, limit: int = 500):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (group_id, limit),
                ).fetchall()
        except Exception:
            return []

    def _clear_short(self, self_id: str, group_id: str):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn, conn:
                conn.execute(
                    "DELETE FROM short_term WHERE group_id = ?", (group_id,)
                )
        except Exception as e:
            logger.warning(f"[绫地宁宁] 清空缓冲失败: {e}")

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
    def _search_long(self, self_id: str, group_id: str, query: str, top_k: int = 5):
        query = (query or "").strip()
        q_norm = _norm(query)
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                rows = conn.execute(
                    "SELECT * FROM long_term WHERE group_id = ? ORDER BY created_at DESC LIMIT 600",
                    (group_id,),
                ).fetchall()
        except Exception as e:
            logger.warning(f"[SmartMemory] 检索失败: {e}")
            return []
        scored = []
        time_level = self._detect_time_intent(query)
        time_since = self._since_ts(time_level) if time_level else 0.0
        for r in rows:
            score = 0
            content = r["content"] or ""
            c_norm = _norm(content)
            # 关键词：原文与简体化后各命中一次（原/简混合也能对上）
            for kw in (r["keywords"] or "").split(","):
                if kw and (kw in query or _norm(kw) in q_norm):
                    score += 2
            # 正文包含提问原文 / 提问的简体形式
            if query and (query in content or q_norm in c_norm):
                score += 3
            # 4-gram 重叠（原文 + 简体化各算一遍）
            grams = self._grams(query) | self._grams(q_norm)
            for gram in grams:
                if gram and (gram in content or gram in c_norm):
                    score += 1
            # 时间意图命中：问“昨天/最近”这类时间词时，时间段内的记忆优先
            if time_since and score == 0 and (r["created_at"] or 0) >= time_since:
                score += 2
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

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
            + ") ORDER BY (content LIKE '%画像%') DESC, created_at DESC LIMIT ?"
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

    def _recent_short_since(self, self_id: str, group_id: str, since_ts: float, n: int = 10):
        try:
            with closing(self._connect(self._db_path(self_id))) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND created_at >= ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (group_id, since_ts, n),
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
            if uid and (not mems or _PROBE_RE.search(query)):
                try:
                    own = self._member_rows(self_id, group_id, uid, uname, 4)
                    if own:
                        dup = {m["id"] for m in (mems or [])}
                        own_lines = []
                        for r in own:
                            if r["id"] in dup:
                                continue
                            when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                            own_lines.append(f"- ({when} {r['user_name']}) {r['content']}")
                        if own_lines:
                            parts.append("【提问者本人档案/记忆，涉及ta自己或你俩关系时以此为准】\n"
                                        + "\n".join(own_lines[:3]))
                except Exception:
                    pass
            # 分层短期记忆：识别提问的时间意图（L2今天/L3最近几天/L4本周）
            level = self._detect_time_intent(query)
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
            elif self.config.get("include_recent", True):
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
