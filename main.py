#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Memory - AstrBot 智能记忆插件
====================================
- 监听群消息：所有消息先进入短期缓存（保留一天）
- 疑似重要的消息由 LLM 判断，重要则摘要进长期记忆
- 群聊触发 LLM 请求时，检索本群相关长期记忆注入 system_prompt
- 按群隔离：群与群之间的记忆互不可见
"""
import asyncio
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType, PermissionType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.components import Plain
from astrbot.core.platform.message_type import MessageType

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:  # 兼容旧版本 AstrBot
    def get_astrbot_data_path() -> str:
        return os.path.realpath(os.path.join(os.getcwd(), "data"))


# 命中这些词的消息才值得交给 LLM 判断（节省 token）
TRIGGER_KEYWORDS = [
    "记住", "记得", "喜欢", "讨厌", "我是", "我叫", "我的名字", "生日",
    "约定", "答应", "打算", "计划", "秘密", "年龄", "考试", "作业",
    "买了", "毕业", "工作", "学校", "最喜欢", "最讨厌", "以后",
    "重要", "别忘了", "提醒", "不要", "希望",
]

JUDGE_SYSTEM_PROMPT = (
    "你是一个群聊记忆筛选器。根据给定的群聊消息，判断它是否值得长期记住。"
    "值得记住的情况举例：个人信息（名字/年龄/生日/喜好/忌口）、重要约定与承诺、"
    "重大事件（考试/毕业/换工作/生病/搬家）、长期计划、群友之间的重要关系。"
    "普通闲聊、吐槽、无信息量的寒暄等属于不重要。"
    "只输出一个 JSON 对象，不要输出任何其他文字，格式："
    '{"important": true或false, "summary": "一句话中文摘要", "keywords": ["关键词1", "关键词2"]}'
    "若 important 为 false，summary 为空字符串，keywords 为空数组。"
)


class SmartMemory(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        data_dir = get_astrbot_data_path()
        self.db_dir = os.path.join(data_dir, "smart_memory")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "memory.db")
        self._init_db()
        self._cleanup_expired()
        # 每日总结定时任务
        self.scheduler = AsyncIOScheduler()
        try:
            hh, mm = str(self.config.get("summary_time", "23:55")).split(":")
            self.scheduler.add_job(
                self._daily_summary_job,
                CronTrigger(hour=int(hh), minute=int(mm)),
                id="stardust_daily_summary",
            )
            self.scheduler.start()
            logger.info(f"[星尘手账] 每日总结定时任务已启动：{hh}:{mm}")
        except Exception as e:
            logger.warning(f"[星尘手账] 定时任务启动失败: {e}")

    # ---------------- 数据库 ----------------
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with closing(self._conn()) as conn, conn:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_long_group ON long_term(group_id)"
            )
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_group ON short_term(group_id)"
            )

    def _cleanup_expired(self):
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "DELETE FROM short_term WHERE expire_at < ?", (time.time(),)
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 清理过期短期记忆失败: {e}")

    # ---------------- 消息监听 ----------------
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        # 忽略 bot 自己的消息，避免自我循环
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        text = event.get_message_str().strip()
        if not text:
            return
        # 插件自身指令不参与记忆
        if text.startswith("/mem"):
            return
        group_id = str(event.get_group_id())
        user_id = str(event.get_sender_id())
        user_name = event.get_sender_name() or user_id

        # 1. 全部消息进短期缓存，保留一天
        self._add_short(group_id, user_id, user_name, text)

        # 2. 疑似重要才交给 LLM 判断（异步，不阻塞回复）
        if self.config.get("judge_enabled", True) and self._should_judge(text):
            asyncio.create_task(
                self._judge_and_store(group_id, user_id, user_name, text)
            )

    def _should_judge(self, text: str) -> bool:
        if len(text) < int(self.config.get("min_len", 10)):
            return False
        return any(kw in text for kw in TRIGGER_KEYWORDS)

    # ---------------- LLM 重要性判断 ----------------
    async def _judge_and_store(
        self, group_id: str, user_id: str, user_name: str, text: str
    ):
        try:
            provider = await self.context.get_using_provider_async()
            if provider is None:
                return
            resp = await provider.text_chat(
                prompt=text,
                system_prompt=JUDGE_SYSTEM_PROMPT,
            )
            out = "".join(
                [c.text for c in resp.result_chain if isinstance(c, Plain)]
            )
            data = self._parse_json(out)
            if not data:
                return
            if data.get("important"):
                summary = (data.get("summary") or text).strip()[:200]
                keywords = data.get("keywords") or []
                self._add_long(group_id, user_id, user_name, summary, keywords, text)
                logger.info(f"[SmartMemory] 已保存长期记忆: {summary}")
        except Exception as e:
            logger.warning(f"[SmartMemory] 重要性判断失败: {e}")

    @staticmethod
    def _parse_json(s: str):
        s = (s or "").strip()
        m = re.search(r"\{.*\}", s, re.S)  # 容忍 ```json 包裹或前后废话
        if m:
            s = m.group(0)
        try:
            return json.loads(s)
        except Exception:
            return None

    # ---------------- 每日总结 ----------------
    async def _daily_summary_job(self):
        try:
            groups = self._groups_today()
            if not groups:
                logger.info("[星尘手账] 今日无消息，跳过总结")
                return
            today = datetime.now().strftime("%m-%d")
            for gid in groups:
                msgs = self._today_msgs(gid, limit=150)
                if not msgs:
                    continue
                text = "\
".join(
                    f"{r['user_name']}: {r['content']}" for r in msgs
                )[:6000]
                provider = await self.context.get_using_provider_async()
                if provider is None:
                    continue
                resp = await provider.text_chat(
                    prompt=text,
                    system_prompt=(
                        "你是群聊日报编辑。根据给定聊天记录提炼最多"
                        f"{int(self.config.get('summary_max_points', 5))}条要点日报，"
                        "覆盖重要话题、群友动态、值得记住的事，忽略灌水闲聊。"
                        '只输出JSON：{"points": ["1. ...", "2. ..."]}'
                    ),
                )
                out = "".join(
                    [c.text for c in resp.result_chain if isinstance(c, Plain)]
                )
                data = self._parse_json(out)
                if not data or not data.get("points"):
                    continue
                points = [str(p).strip() for p in data["points"]][
                    : int(self.config.get("summary_max_points", 5))
                ]
                summary = f"【日报 {today}】" + "；".join(points)
                self._add_long(
                    gid, "system", "星尘手账日报", summary,
                    ["日报", "总结"], f"每日总结 {today}",
                )
                logger.info(f"[星尘手账] 群 {gid} 日报已生成")
        except Exception as e:
            logger.warning(f"[星尘手账] 每日总结失败: {e}")

    def _groups_today(self):
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT group_id FROM short_term WHERE created_at >= ?",
                    (time.time() - 86400,),
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _today_msgs(self, group_id: str, limit: int = 150):
        try:
            with closing(self._conn()) as conn:
                return conn.execute(
                    "SELECT * FROM short_term WHERE group_id = ? AND created_at >= ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (group_id, time.time() - 86400, limit),
                ).fetchall()
        except Exception:
            return []

    # ---------------- 存储 ----------------
    def _add_short(self, group_id, user_id, user_name, text):
        now = time.time()
        try:
            with closing(self._conn()) as conn, conn:
                conn.execute(
                    "INSERT INTO short_term (group_id, user_id, user_name, content, created_at, expire_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (group_id, user_id, user_name, text[:500], now, now + 86400),
                )
        except Exception as e:
            logger.warning(f"[SmartMemory] 短期记忆写入失败: {e}")

    def _add_long(self, group_id, user_id, user_name, summary, keywords, raw):
        try:
            with closing(self._conn()) as conn, conn:
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
    def _search_long(self, group_id: str, query: str, top_k: int = 5):
        query = (query or "").strip()
        try:
            with closing(self._conn()) as conn:
                rows = conn.execute(
                    "SELECT * FROM long_term WHERE group_id = ? ORDER BY created_at DESC LIMIT 300",
                    (group_id,),
                ).fetchall()
        except Exception as e:
            logger.warning(f"[SmartMemory] 检索失败: {e}")
            return []
        scored = []
        for r in rows:
            score = 0
            for kw in (r["keywords"] or "").split(","):
                if kw and kw in query:
                    score += 2
            content = r["content"] or ""
            if query and query in content:
                score += 3
            for gram in self._grams(query):
                if gram and gram in content:
                    score += 1
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    @staticmethod
    def _grams(s: str, n: int = 4):
        s = re.sub(r"\s+", "", s or "")
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

    def _recent_short(self, group_id: str, n: int = 10):
        try:
            with closing(self._conn()) as conn:
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
            if event.get_message_type() != MessageType.GROUP_MESSAGE:
                return
            group_id = str(event.get_group_id())
            query = event.get_message_str().strip()
            if not query:
                return
            parts = []
            # 相关长期记忆
            top_k = int(self.config.get("top_k", 5))
            mems = self._search_long(group_id, query, top_k)
            if mems:
                lines = []
                for r in mems:
                    when = datetime.fromtimestamp(r["created_at"]).strftime("%m-%d")
                    lines.append(f"- ({when} {r['user_name']}) {r['content']}")
                parts.append("【记忆片段，来自本群历史消息，可参考但勿编造】\n" + "\n".join(lines))
            # 最近短期消息
            if self.config.get("include_recent", True):
                n = int(self.config.get("recent_count", 5))
                recent = self._recent_short(group_id, n)
                if recent:
                    lines = [
                        f"- {r['user_name']}: {r['content']}" for r in reversed(recent)
                    ]
                    parts.append("【最近消息，可能与本条相关】\n" + "\n".join(lines))
            if parts:
                req.system_prompt = (req.system_prompt or "").rstrip() + "\n\n" + "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"[SmartMemory] 注入记忆失败: {e}")

    # ---------------- 指令 ----------------
    @filter.command("mem", alias={"memory"})
    async def mem(self, event: AstrMessageEvent):
        args = (event.message_str or "").split()
        sub = args[1].strip().lower() if len(args) > 1 else "help"
        group_id = str(event.get_group_id())
        user_id = str(event.get_sender_id())

        if sub == "list":
            n = int(args[2]) if len(args) > 2 and args[2].isdigit() else 10
            try:
                with closing(self._conn()) as conn:
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
            recent = self._recent_short(group_id, n)
            if not recent:
                yield event.plain_result("本群最近没有短期消息记录～")
                return
            lines = [f"本群最近 {len(recent)} 条消息（一天内有效）："]
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
                with closing(self._conn()) as conn:
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

        if sub == "stat":
            try:
                with closing(self._conn()) as conn:
                    long_n = conn.execute(
                        "SELECT COUNT(*) FROM long_term WHERE group_id = ?", (group_id,)
                    ).fetchone()[0]
                    short_n = conn.execute(
                        "SELECT COUNT(*) FROM short_term WHERE group_id = ? AND expire_at > ?",
                        (group_id, time.time()),
                    ).fetchone()[0]
                yield event.plain_result(
                    f"本群记忆统计：长期 {long_n} 条，短期（一天内）{short_n} 条"
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
