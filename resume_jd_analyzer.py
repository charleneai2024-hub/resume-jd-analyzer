"""Five-stage chain: resume <-> JD fit analyzer for an AI PM role.

Ring 1 (LLM): extract requirement items from the JD, each tagged hard/soft.
Ring 2 (LLM, CoT + evidence): for every requirement, pull the supporting
    sentence from the resume first, then judge met/partial/missing.
Ring 3 (pure Python): group ring 2's verdicts, count them, compute a rough
    fit score.
Ring 4 (LLM): turn the partial/missing items into concrete, encouraging
    Chinese suggestions.
Ring 5 (LLM): QC pass - checks ring 4's suggestions against ring 2's
    evidence for fabrication and vagueness, without rewriting them.
"""

import json
import sys

from llm_client import call_llm_messages

# 示例数据:以下JD为虚构的通用AI产品经理岗位描述,不对应任何真实招聘信息,
# 仅用于演示链路效果。
JD_TEXT = """We are looking for a data-driven AI Product Manager to lead the
design, launch, and iteration of AI-powered features for our platform. You
will work at the intersection of business, design, and engineering,
translating AI/ML capabilities into products that are useful, trustworthy,
and easy to adopt.

Key Responsibilities
- Define product vision and roadmap for AI initiatives, aligned with
  business goals.
- Own the end-to-end lifecycle of AI features: problem framing, requirement
  definition, prioritization, launch, and iteration.
- Partner closely with data scientists and engineers to translate model
  capabilities and constraints into product decisions.
- Define evaluation metrics and quality guardrails for AI features (e.g.
  accuracy, latency, failure rate) and monitor them post-launch.
- Conduct user research to identify problems where AI meaningfully
  outperforms existing solutions.
- Write clear PRDs and manage cross-functional stakeholder alignment.

Required Qualifications
- 3+ years of product management experience, including 1+ year on an
  AI/ML or data-intensive product.
- Working knowledge of modern AI concepts (e.g. LLMs, embeddings, model
  evaluation) sufficient to discuss trade-offs with an engineering team.
- Strong data analysis skills; comfortable defining and interpreting
  product/model metrics.
- Excellent written communication skills, including PRD writing.
- Experience working in Agile/Scrum environments.

Preferred Qualifications
- Bachelor's degree in Computer Science, Data Science, or a related field.
- Experience shipping a generative AI or conversational AI feature to
  production.
- Familiarity with prompt engineering or LLM evaluation workflows."""

# 示例数据:以下简历为虚构人物"陈明"的杜撰经历,姓名、公司、数据均为示例,
# 不含任何真实联系方式,仅用于演示链路效果。
RESUME_TEXT = """陈明 (Ming Chen)
产品经理

EXPERIENCE

Senior Product Analyst, Horizon Retail Co.
Mar 2021 - Present
- Owned the metrics framework for the company's mobile app, covering
  activation, retention, and conversion; built weekly dashboards used by
  product and growth teams.
- Led root-cause analysis on a 15% drop in checkout conversion, identified
  a broken coupon-code flow, and worked with engineering to ship a fix
  within one sprint.
- Partnered with engineering and design to prioritize the quarterly
  roadmap; wrote requirement docs for 10+ shipped features.
- Ran A/B tests on onboarding flow changes, resulting in a 6% lift in
  Day-7 retention.
- Presented quarterly business reviews to senior stakeholders, translating
  data findings into actionable recommendations.

Product Analyst, Northgate Logistics
Jul 2018 - Feb 2021
- Built and maintained SQL-based reporting for warehouse operations,
  reducing manual reporting time by 40%.
- Collaborated with operations and engineering teams to define KPIs for a
  new route-planning tool.
- Supported a pilot chatbot project used for internal customer service
  ticket triage; worked with a vendor's rule-based bot platform to define
  intents and escalation logic.

SKILLS
Programming: SQL, Python (basic scripting)
Data & Visualization: Tableau, Excel, Google Analytics
Product: PRD writing, Agile/Scrum, A/B testing, roadmap prioritization
Soft Skills: cross-functional communication, stakeholder presentations

EDUCATION
Bachelor of Business Administration, Analytics concentration
Riverdale State University, 2014 - 2018

CERTIFICATIONS
Certified Scrum Product Owner (CSPO)
"""


def _strip_code_fence(text: str) -> str:
    """Haiku sometimes wraps JSON in ```json ... ``` despite instructions
    not to; strip that before parsing."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


def _parse_json(raw_output: str, stage_label: str):
    try:
        return json.loads(_strip_code_fence(raw_output))
    except json.JSONDecodeError as error:
        print(f"[{stage_label}] 模型输出无法解析为JSON,原始输出如下:")
        print(raw_output)
        print(f"[{stage_label}] JSON解析错误: {error}")
        sys.exit(1)


def ring1_extract_requirements(jd_text: str) -> str:
    system = (
        "You are a precise information extraction engine. Extract every "
        "distinct requirement item for a Product Manager role from the job "
        "description the user provides. Classify each item's type as "
        '"hard" (concrete/verifiable, e.g. years of experience, specific '
        'tools, degrees) or "soft" (interpersonal/communication/soft '
        "skills). Output only a JSON array, for example: "
        '[{"requirement":"2+ years PM experience","type":"hard"}]. '
        "Do not include any explanation, markdown code fences, or any "
        "other text."
    )
    return call_llm_messages(
        messages=[{"role": "user", "content": jd_text}],
        system=system,
        max_tokens=1024,
    )


def ring2_match_requirements(requirements_json: str, resume_text: str) -> str:
    system = (
        "You are a meticulous resume screener. The user will give you a "
        "JSON array of job requirements and a resume text. For EACH "
        "requirement, you must reason in this order: first locate the "
        "exact sentence(s) in the resume that support your judgment (the "
        '"evidence" field); if no such sentence exists, evidence must be '
        'an empty string "". Only after identifying evidence may you decide '
        'the "verdict": "met" (clearly satisfied by the evidence), '
        '"partial" (some relevant evidence but incomplete or indirect), or '
        '"missing" (no supporting evidence found). You must never guess or '
        "infer a verdict that is not grounded in actual resume text - if "
        "there is no evidence, the verdict cannot be \"met\". Also give a "
        'one-sentence "reason" explaining the verdict. Output only a JSON '
        "array, for example: "
        '[{"requirement":"...","verdict":"partial","evidence":"简历原句",'
        '"reason":"..."}]. Do not include any explanation outside the '
        "JSON, markdown code fences, or any other text."
    )
    user_content = (
        f"职位要求列表(JSON):\n{requirements_json}\n\n简历原文:\n{resume_text}"
    )
    return call_llm_messages(
        messages=[{"role": "user", "content": user_content}],
        system=system,
        max_tokens=2048,
    )


def ring3_summarize(matches: list[dict]) -> dict:
    groups = {"met": [], "partial": [], "missing": []}
    for item in matches:
        verdict = item.get("verdict", "missing")
        groups.setdefault(verdict, []).append(item)

    total = len(matches)
    met_count = len(groups["met"])
    fit_score = met_count / total if total else 0.0

    print(f"要求总数: {total}")
    print(f"met (满足): {len(groups['met'])}")
    print(f"partial (部分满足): {len(groups['partial'])}")
    print(f"missing (缺失): {len(groups['missing'])}")
    print(f"粗略契合度 (met/总数): {fit_score:.0%}")

    return {"groups": groups, "fit_score": fit_score, "total": total}


def ring4_generate_advice(groups: dict) -> str:
    gap_items = groups["partial"] + groups["missing"]
    system = (
        "你是一位耐心、专业的职业发展顾问。用户会给你一份JSON数组,里面是简历"
        "相对于岗位要求中『部分满足』或『缺失』的条目,每条包含requirement、"
        "verdict、evidence、reason。请针对每一条要求,给出具体、可执行的改进"
        "建议(例如补充哪类项目经验、量化哪些数据、在简历中如何措辞等),"
        "每条建议要明确对应到具体的requirement。语气要礼貌、鼓励,避免空泛"
        "的套话。使用中文输出,可以用自然语言分条陈述,不需要输出JSON。"
        "改进建议只能基于简历中已经存在的信息和经历。严禁假设、脑补或虚构"
        "任何简历未明确提及的项目、经历或成果。如果某条要求在简历中缺乏"
        "任何可利用的基础,就诚实说明'简历中暂无相关基础,建议从零积累',"
        "而不是编造细节。"
    )
    user_content = "待改进条目(JSON):\n" + json.dumps(gap_items, ensure_ascii=False)
    return call_llm_messages(
        messages=[{"role": "user", "content": user_content}],
        system=system,
        max_tokens=1500,
    )


def ring5_quality_check(matches: list[dict], advice_text: str) -> str:
    system = (
        "你是一名严格的质检员,负责审核『简历改进建议』的质量。用户会给你两"
        "部分内容:(1)简历与岗位要求的逐条比对结果(JSON,含requirement、"
        "verdict、evidence、reason);(2)基于这些比对结果生成的改进建议"
        "(自然语言)。请你逐条核查建议,检查:(a)建议中提到的信息是否能在"
        "比对结果的evidence中找到依据,是否存在编造简历中不存在的经历或数据;"
        "(b)建议是否具体可落地,而不是『多积累经验』『加强沟通能力』这类空话。"
        "对每一条建议给出判断,若发现问题(编造或空泛),明确标记并说明原因;"
        "不需要重写建议本身。用中文输出一份结构清晰的质检报告。"
    )
    user_content = (
        "逐条比对结果(JSON):\n"
        + json.dumps(matches, ensure_ascii=False)
        + "\n\n改进建议:\n"
        + advice_text
    )
    return call_llm_messages(
        messages=[{"role": "user", "content": user_content}],
        system=system,
        max_tokens=1500,
    )


def main() -> None:
    print("=" * 60)
    print("环1: 从JD提取要求条目")
    print("=" * 60)
    ring1_output = ring1_extract_requirements(JD_TEXT)
    print(ring1_output)
    requirements = _parse_json(ring1_output, "环1")

    print()
    print("=" * 60)
    print("环2: 逐条语义比对(先找证据,再下判定)")
    print("=" * 60)
    ring2_output = ring2_match_requirements(json.dumps(requirements, ensure_ascii=False), RESUME_TEXT)
    print(ring2_output)
    matches = _parse_json(ring2_output, "环2")

    print()
    print("=" * 60)
    print("环3: 归类统计")
    print("=" * 60)
    summary = ring3_summarize(matches)

    print()
    print("=" * 60)
    print("环4: 针对partial/missing生成改进建议")
    print("=" * 60)
    advice_text = ring4_generate_advice(summary["groups"])
    print(advice_text)

    print()
    print("=" * 60)
    print("环5: 质检建议(不编造、不空泛)")
    print("=" * 60)
    qc_report = ring5_quality_check(matches, advice_text)
    print(qc_report)


if __name__ == "__main__":
    main()
