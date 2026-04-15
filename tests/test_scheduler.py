from axis.scheduler import Scheduler, SchedulerConfig
from axis.types import Stakes, Task


def test_allocate_splits_into_four_sub_budgets():
    s = Scheduler()
    t = Task(input="hi", task_class="general", stakes=Stakes.MEDIUM, complexity=0.5)
    b = s.allocate(t)
    assert b.total_tokens > 0
    assert (
        b.planning_tokens
        + b.execution_tokens
        + b.verification_tokens
        + b.buffer_tokens
        == b.total_tokens
    )
    # 15% planning split from spec §7.1
    assert b.planning_tokens == int(b.total_tokens * 0.15)


def test_stakes_scale_budget_monotonically():
    s = Scheduler()
    low = s.allocate(Task(input="hi", stakes=Stakes.LOW))
    med = s.allocate(Task(input="hi", stakes=Stakes.MEDIUM))
    high = s.allocate(Task(input="hi", stakes=Stakes.HIGH))
    crit = s.allocate(Task(input="hi", stakes=Stakes.CRITICAL))
    assert low.total_tokens < med.total_tokens < high.total_tokens <= crit.total_tokens


def test_hard_ceiling_enforced():
    s = Scheduler()
    t = Task(
        input="hi",
        task_class="legal_reasoning",
        stakes=Stakes.CRITICAL,
        complexity=1.0,
    )
    b = s.allocate(t)
    assert b.total_tokens <= s.config.hard_ceiling


def test_complexity_scales_budget():
    s = Scheduler()
    easy = s.allocate(Task(input="hi", complexity=0.0))
    hard = s.allocate(Task(input="hi", complexity=1.0))
    assert hard.total_tokens > easy.total_tokens


def test_critical_always_verifies():
    s = Scheduler()
    g = s.gates(Task(input="x", stakes=Stakes.CRITICAL))
    assert g.verification is True


def test_low_stakes_skips_verification_in_known_domain():
    s = Scheduler()
    self_model = {"capabilities": {"chat": {"win_rate": 0.95}}}
    g = s.gates(
        Task(input="x", task_class="chat", stakes=Stakes.LOW),
        self_model=self_model,
        novelty=0.0,
    )
    assert g.verification is False


def test_retrieval_gate_opens_on_novelty():
    s = Scheduler()
    g_low = s.gates(Task(input="x", stakes=Stakes.MEDIUM), novelty=0.1)
    g_high = s.gates(Task(input="x", stakes=Stakes.MEDIUM), novelty=0.9)
    assert g_high.retrieval is True
    # Low novelty + unknown domain confidence (0.5) still opens retrieval
    # because domain_confidence 0.5 < threshold 0.7.
    assert g_low.retrieval is True


def test_retrieval_closed_on_high_domain_confidence_and_low_novelty():
    s = Scheduler()
    self_model = {"capabilities": {"chat": {"win_rate": 0.92}}}
    g = s.gates(
        Task(input="x", task_class="chat", stakes=Stakes.MEDIUM),
        self_model=self_model,
        novelty=0.1,
    )
    assert g.retrieval is False


def test_exploration_budget_allocated_when_strategy_weak():
    s = Scheduler()
    self_model = {"capabilities": {"chat": {"win_rate": 0.4}}}
    g = s.gates(
        Task(input="x", task_class="chat", stakes=Stakes.LOW),
        self_model=self_model,
    )
    assert g.exploration_fraction > 0.0


def test_should_continue_budget_and_marginal_gain():
    s = Scheduler()
    b = s.allocate(Task(input="hi"))
    # Plenty of budget, gain > cost → continue
    assert s.should_continue(0, b, 0.5, 0.1, 0) is True
    # Budget exhausted → stop
    assert s.should_continue(b.total_tokens, b, 0.5, 0.1, 0) is False
    # Marginal gain below cost → stop
    assert s.should_continue(0, b, 0.1, 0.5, 0) is False


def test_should_continue_respects_deadline():
    s = Scheduler()
    t = Task(input="hi", deadline_ms=100)
    b = s.allocate(t)
    assert s.should_continue(0, b, 0.5, 0.1, 50) is True
    assert s.should_continue(0, b, 0.5, 0.1, 150) is False


def test_config_respects_override():
    cfg = SchedulerConfig(task_budgets={"general": 1000})
    s = Scheduler(cfg)
    b = s.allocate(Task(input="hi", stakes=Stakes.MEDIUM, complexity=0.5))
    # base 1000 * complexity 1.1 * stakes 1.0 = 1100
    assert b.total_tokens == 1100
