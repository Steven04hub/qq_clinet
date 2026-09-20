import pytest

from app.services.rate_limit import RateLimiter


@pytest.fixture
def clock():
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    return Clock()


def test_first_allowed_and_immediate_second_denied(clock):
    limiter = RateLimiter(5, clock=clock)
    assert limiter.allow(100)
    assert not limiter.allow(100)


def test_cooldown_expires_at_exact_boundary_without_denials_extending_it(clock):
    limiter = RateLimiter(5, clock=clock)
    assert limiter.allow(100)
    clock.now = 4.999
    assert not limiter.allow(100)
    clock.now = 5.0
    assert limiter.allow(100)
    assert not limiter.allow(100)


def test_users_have_independent_cooldowns(clock):
    limiter = RateLimiter(5, clock=clock)
    assert limiter.allow(100)
    clock.now = 3
    assert limiter.allow(200)
    clock.now = 5
    assert limiter.allow(100)
    assert not limiter.allow(200)
    clock.now = 8
    assert limiter.allow(200)


def test_capacity_rejects_new_users_without_evicting_active_cooldowns(clock):
    limiter = RateLimiter(5, clock=clock, max_users=2)
    assert limiter.allow(100)
    clock.now = 1
    assert limiter.allow(200)
    assert not limiter.allow(300)
    assert not limiter.allow(100)
    clock.now = 5
    assert limiter.allow(300)
    assert not limiter.allow(200)
    clock.now = 6
    assert limiter.allow(400)
    assert not limiter.allow(300)


def test_new_user_burst_is_bounded_and_expiration_recovers_capacity(clock):
    limiter = RateLimiter(5, clock=clock, max_users=3)
    assert sum(limiter.allow(user_id) for user_id in range(1000)) == 3
    clock.now = 5
    assert sum(limiter.allow(user_id) for user_id in range(1000, 2000)) == 3


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan"), True])
def test_invalid_cooldown_is_rejected(interval):
    with pytest.raises(ValueError):
        RateLimiter(interval)


@pytest.mark.parametrize("max_users", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected(max_users):
    with pytest.raises(ValueError):
        RateLimiter(5, max_users=max_users)
